"""Trainable transformations and the shared six-model classical benchmark."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

import config as C


class VisualVocabulary(BaseEstimator, TransformerMixin):
    def __init__(self, words=32, representation="histogram", seed=42):
        self.words = words
        self.representation = representation
        self.seed = seed

    @staticmethod
    def _arrays(X):
        return X.iloc[:, 0].to_list()

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        arrays = self._arrays(X)
        self.descriptor_width_ = arrays[0].shape[1]
        sampled = [
            values[rng.choice(len(values), min(32, len(values)), replace=False)]
            for values in arrays
            if len(values)
        ]
        self.cluster_ = None
        if sampled:
            stack = np.vstack(sampled)
            if len(stack) > 24000:
                stack = stack[rng.choice(len(stack), 24000, replace=False)]
            clusters = min(self.words, len(stack), len(np.unique(stack, axis=0)))
            self.cluster_ = MiniBatchKMeans(clusters, random_state=self.seed, n_init=3, batch_size=512)
            with threadpool_limits(limits=1):
                self.cluster_.fit(stack)
        return self

    def transform(self, X):
        descriptor_width = self.descriptor_width_
        width = 0
        if self.representation in ("histogram", "both"):
            width += self.words
        if self.representation in ("vlad", "both"):
            width += self.words * descriptor_width
        arrays = self._arrays(X)
        output = np.zeros((len(arrays), width), dtype=float)
        if self.cluster_ is None:
            return output
        centers = self.cluster_.cluster_centers_
        for index, values in enumerate(arrays):
            if not len(values):
                continue
            words = self.cluster_.predict(values)
            offset = 0
            if self.representation in ("histogram", "both"):
                output[index, : self.words] = np.bincount(words, minlength=self.words) / len(words)
                offset = self.words
            if self.representation in ("vlad", "both"):
                residuals = np.zeros((self.words, descriptor_width), dtype=float)
                np.add.at(residuals, words, values - centers[words])
                residuals = np.sign(residuals) * np.sqrt(abs(residuals))
                output[index, offset:] = (residuals / max(np.linalg.norm(residuals), 1e-12)).ravel()
        return output


class ClippedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, components=32, seed=42):
        self.components = components
        self.seed = seed

    def fit(self, X, y=None):
        count = min(self.components, X.shape[1], max(1, X.shape[0] - 1))
        self.pca_ = PCA(count, svd_solver="randomized", random_state=self.seed).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def classifier(name, min_fold_size):
    seed = C.SEED
    if name == "logistic":
        return LogisticRegression(max_iter=1500, C=1, random_state=seed), {"C": [0.1, 1]}
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {"shrinkage": ["auto", 0.3]}
    if name == "svm":
        return SVC(C=3, kernel="rbf"), {"C": [1, 10], "gamma": ["scale"]}
    if name == "random_forest":
        model = RandomForestClassifier(n_estimators=160, max_features="sqrt", random_state=seed, n_jobs=1)
        return model, {"min_samples_leaf": [1, 3]}
    if name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=160,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
        )
        return model, {"max_depth": [3, 5]}
    if name == "knn":
        neighbors = sorted({min(k, min_fold_size) for k in (3, 7)})
        return KNeighborsClassifier(n_neighbors=neighbors[0], weights="distance"), {"n_neighbors": neighbors}
    if name == "tree":
        return DecisionTreeClassifier(max_depth=3, random_state=seed), {"max_depth": [3]}
    if name == "dummy":
        return DummyClassifier(strategy="prior"), {}
    raise ValueError(f"Unknown model {name!r}. Check MODELS in config.py.")


def training_pipeline(name, numeric, vocabulary, pca, min_fold_size):
    estimator, parameters = classifier(name, min_fold_size)
    grid = {f"model__{key}": options for key, options in parameters.items()} if C.TUNE_MODELS else {}
    transforms = []
    if numeric:
        numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if pca:
            numeric_steps.extend(
                [("scale", StandardScaler()), ("pca", ClippedPCA(C.PCA_COMPONENTS[0], C.SEED))]
            )
            grid["features__numeric__pca__components"] = list(C.PCA_COMPONENTS)
        transforms.append(("numeric", Pipeline(numeric_steps), numeric))
    if vocabulary:
        words = VisualVocabulary(C.VOCABULARY_SIZES[0], vocabulary, C.SEED)
        transforms.append(("vocabulary", words, ["visual_patches"]))
        grid["features__vocabulary__words"] = list(C.VOCABULARY_SIZES)
    steps = []
    if transforms:
        steps.append(("features", ColumnTransformer(transforms)))
        if name in {"svm", "logistic", "lda", "knn"}:
            steps.append(("scale", StandardScaler()))
        wide_input = vocabulary or (not pca and len(numeric) > (C.LDA_MAX_COMPONENTS or 0))
        if name == "lda" and C.LDA_MAX_COMPONENTS and wide_input:
            steps.append(("lda_pca", ClippedPCA(C.LDA_MAX_COMPONENTS, C.SEED)))
    steps.append(("model", estimator))
    return Pipeline(steps), grid
