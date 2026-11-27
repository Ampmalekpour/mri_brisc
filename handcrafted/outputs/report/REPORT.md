# BRISC handcrafted feature study

The primary test excludes official test cases in a duplicate or patient group that touches training.
The official split is also reported for comparability. Model selection uses grouped folds from training.
Tumor3 includes glioma, meningioma and pituitary cases with usable annotations.
Image4 uses image-derived inputs and includes the no-tumor class.

## Indexed sample counts

| split | label | images |
| --- | --- | --- |
| test | glioma | 254 |
| test | meningioma | 306 |
| test | no_tumor | 140 |
| test | pituitary | 300 |
| train | glioma | 1147 |
| train | meningioma | 1329 |
| train | no_tumor | 1067 |
| train | pituitary | 1457 |
| val | glioma | 254 |
| val | meningioma | 306 |
| val | no_tumor | 140 |
| val | pituitary | 300 |

## Classical models selected by cross-validation

| experiment | description | cohort | model | cv_balanced_accuracy | images | accuracy | balanced_accuracy | macro_f1 | macro_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E20 | All handcrafted features | tumor3 | xgboost | 0.987 | 756 | 0.974 | 0.972 | 0.972 | 0.999 |
| E19 | Medically motivated features | tumor3 | xgboost | 0.979 | 756 | 0.955 | 0.952 | 0.953 | 0.996 |
| E17 | Tumor features and local context | tumor3 | xgboost | 0.978 | 756 | 0.954 | 0.950 | 0.952 | 0.994 |
| F21 | F21 independently | tumor3 | svm | 0.967 | 756 | 0.951 | 0.950 | 0.949 | 0.986 |
| E16 | Intrinsic tumor features and position | tumor3 | xgboost | 0.963 | 756 | 0.930 | 0.928 | 0.929 | 0.986 |
| F24 | F24 independently | tumor3 | svm | 0.959 | 756 | 0.964 | 0.961 | 0.962 | 0.988 |
| 4C04 | Four-class all image-only features | image4 | xgboost | 0.958 | 896 | 0.919 | 0.922 | 0.915 | 0.990 |
| E13H | Head-restricted peritumoral rings | tumor3 | svm | 0.950 | 756 | 0.942 | 0.939 | 0.939 | 0.982 |
| E13 | Peritumoral rings | tumor3 | svm | 0.950 | 756 | 0.943 | 0.940 | 0.941 | 0.983 |
| E13U | Unrestricted peritumoral rings | tumor3 | svm | 0.949 | 756 | 0.942 | 0.939 | 0.940 | 0.981 |
| 4C03 | Four-class whole MRI | image4 | xgboost | 0.943 | 896 | 0.922 | 0.926 | 0.920 | 0.992 |
| F23 | F23 independently | tumor3 | svm | 0.942 | 756 | 0.910 | 0.907 | 0.906 | 0.976 |
| 4C02 | Four-class acquisition and background | image4 | xgboost | 0.922 | 896 | 0.886 | 0.892 | 0.884 | 0.972 |
| E15 | Intrinsic tumor appearance, size and shape | tumor3 | xgboost | 0.918 | 756 | 0.843 | 0.838 | 0.841 | 0.956 |
| E14 | Tumor appearance in centered views | tumor3 | svm | 0.917 | 756 | 0.869 | 0.867 | 0.868 | 0.959 |
| E10 | Texture | tumor3 | svm | 0.914 | 756 | 0.872 | 0.868 | 0.872 | 0.961 |
| E02 | Acquisition and background | tumor3 | xgboost | 0.904 | 756 | 0.881 | 0.875 | 0.873 | 0.970 |
| F02 | F02 independently | tumor3 | xgboost | 0.903 | 756 | 0.886 | 0.881 | 0.879 | 0.969 |
| F15 | F15 independently | tumor3 | svm | 0.898 | 756 | 0.862 | 0.858 | 0.862 | 0.948 |
| T32 | Texture with 32 gray levels | tumor3 | svm | 0.898 | 756 | 0.862 | 0.858 | 0.862 | 0.948 |
| E11 | Internal lesion architecture | tumor3 | svm | 0.896 | 756 | 0.858 | 0.855 | 0.859 | 0.948 |
| E18 | Distant tissue and background | tumor3 | xgboost | 0.895 | 756 | 0.869 | 0.865 | 0.861 | 0.968 |
| T16 | Texture with 16 gray levels | tumor3 | svm | 0.894 | 756 | 0.854 | 0.852 | 0.854 | 0.946 |
| T64 | Texture with 64 gray levels | tumor3 | svm | 0.893 | 756 | 0.848 | 0.842 | 0.847 | 0.943 |
| F13 | F13 independently | tumor3 | xgboost | 0.891 | 756 | 0.865 | 0.865 | 0.864 | 0.963 |
| E09 | Reference-normalized intensity | tumor3 | xgboost | 0.891 | 756 | 0.865 | 0.865 | 0.864 | 0.963 |
| V04 | HOG of the full MRI | tumor3 | svm | 0.860 | 756 | 0.861 | 0.852 | 0.851 | 0.947 |
| F19 | F19 independently | tumor3 | xgboost | 0.860 | 756 | 0.771 | 0.774 | 0.772 | 0.920 |
| E06 | Location and size | tumor3 | xgboost | 0.859 | 756 | 0.880 | 0.870 | 0.871 | 0.969 |
| F17 | F17 independently | tumor3 | svm | 0.848 | 756 | 0.825 | 0.827 | 0.827 | 0.931 |
| E07 | Mask-only location, size and shape | tumor3 | xgboost | 0.845 | 756 | 0.718 | 0.683 | 0.674 | 0.855 |
| E00 | Small geometry and plane probe | tumor3 | random_forest | 0.845 | 756 | 0.791 | 0.769 | 0.769 | 0.910 |
| V03 | HOG of centered local context | tumor3 | svm | 0.828 | 756 | 0.804 | 0.796 | 0.795 | 0.922 |
| F20 | F20 independently | tumor3 | svm | 0.826 | 756 | 0.754 | 0.745 | 0.744 | 0.890 |
| E12 | Boundary appearance | tumor3 | svm | 0.826 | 756 | 0.754 | 0.745 | 0.744 | 0.890 |
| F14 | F14 independently | tumor3 | random_forest | 0.818 | 756 | 0.726 | 0.733 | 0.729 | 0.892 |
| E08 | Raw intensity | tumor3 | xgboost | 0.817 | 756 | 0.737 | 0.743 | 0.739 | 0.897 |
| F22 | F22 independently | tumor3 | random_forest | 0.810 | 756 | 0.803 | 0.805 | 0.800 | 0.919 |
| F12 | F12 independently | tumor3 | svm | 0.810 | 756 | 0.713 | 0.718 | 0.715 | 0.868 |
| F18 | F18 independently | tumor3 | svm | 0.810 | 756 | 0.790 | 0.784 | 0.787 | 0.904 |
| F05 | F05 independently | tumor3 | random_forest | 0.795 | 756 | 0.824 | 0.813 | 0.810 | 0.940 |
| F25 | F25 independently | tumor3 | svm | 0.784 | 756 | 0.761 | 0.754 | 0.751 | 0.893 |
| F11 | F11 independently | tumor3 | svm | 0.774 | 756 | 0.774 | 0.761 | 0.759 | 0.888 |
| F16 | F16 independently | tumor3 | svm | 0.762 | 756 | 0.743 | 0.748 | 0.746 | 0.873 |
| F04 | F04 independently | tumor3 | random_forest | 0.762 | 756 | 0.757 | 0.745 | 0.744 | 0.910 |
| V06 | Lesion VLAD | tumor3 | xgboost | 0.741 | 756 | 0.665 | 0.668 | 0.671 | 0.856 |
| P02 | PCA of centered lesion appearance | tumor3 | xgboost | 0.737 | 756 | 0.640 | 0.650 | 0.643 | 0.825 |
| V05 | Lesion visual word histogram | tumor3 | xgboost | 0.725 | 756 | 0.651 | 0.657 | 0.657 | 0.843 |
| E05 | Shape and topology | tumor3 | xgboost | 0.725 | 756 | 0.536 | 0.508 | 0.494 | 0.687 |
| F06 | F06 independently | tumor3 | svm | 0.716 | 756 | 0.675 | 0.655 | 0.649 | 0.829 |
| E04 | Lesion size | tumor3 | svm | 0.716 | 756 | 0.675 | 0.655 | 0.649 | 0.829 |
| F01 | F01 independently | tumor3 | xgboost | 0.708 | 756 | 0.578 | 0.545 | 0.534 | 0.759 |
| V02 | HOG of centered lesion appearance | tumor3 | svm | 0.678 | 756 | 0.636 | 0.638 | 0.634 | 0.811 |
| E03 | Normalized centroid only | tumor3 | knn | 0.665 | 756 | 0.661 | 0.645 | 0.641 | 0.831 |
| E04N | Normalized lesion size | tumor3 | random_forest | 0.662 | 756 | 0.676 | 0.663 | 0.659 | 0.842 |
| F09 | F09 independently | tumor3 | xgboost | 0.649 | 756 | 0.443 | 0.418 | 0.397 | 0.606 |
| F07 | F07 independently | tumor3 | svm | 0.621 | 756 | 0.512 | 0.495 | 0.494 | 0.676 |
| F08 | F08 independently | tumor3 | svm | 0.594 | 756 | 0.418 | 0.410 | 0.404 | 0.605 |
| F10 | F10 independently | tumor3 | svm | 0.575 | 756 | 0.450 | 0.422 | 0.410 | 0.593 |
| P01 | PCA of centered masks | tumor3 | svm | 0.571 | 756 | 0.480 | 0.479 | 0.477 | 0.670 |
| V01 | HOG of centered mask | tumor3 | svm | 0.545 | 756 | 0.437 | 0.440 | 0.431 | 0.639 |
| F03 | F03 independently | tumor3 | lda | 0.335 | 756 | 0.390 | 0.333 | 0.187 | 0.476 |
| E01 | Plane only | tumor3 | lda | 0.335 | 756 | 0.390 | 0.333 | 0.187 | 0.476 |
| 4C01 | Four-class plane | image4 | lda | 0.253 | 896 | 0.329 | 0.250 | 0.124 | 0.484 |

Class-wise precision, recall, and F1 are in `classical_selected_per_class.csv`.

## Paired changes after adding feature groups

| before | after | model | change | cv_balanced_accuracy_gain | balanced_accuracy_difference |
| --- | --- | --- | --- | --- | --- |
| E15 | E16 | logistic | Adding tumor position | 0.043 | 0.059 |
| E15 | E16 | lda | Adding tumor position | 0.035 | 0.069 |
| E15 | E16 | svm | Adding tumor position | 0.037 | 0.054 |
| E15 | E16 | random_forest | Adding tumor position | 0.044 | 0.087 |
| E15 | E16 | xgboost | Adding tumor position | 0.045 | 0.090 |
| E15 | E16 | knn | Adding tumor position | 0.038 | 0.070 |
| E16 | E17 | logistic | Adding local tissue context | 0.029 | 0.038 |
| E16 | E17 | lda | Adding local tissue context | 0.042 | 0.030 |
| E16 | E17 | svm | Adding local tissue context | 0.020 | 0.032 |
| E16 | E17 | random_forest | Adding local tissue context | 0.029 | 0.029 |
| E16 | E17 | xgboost | Adding local tissue context | 0.015 | 0.022 |
| E16 | E17 | knn | Adding local tissue context | 0.048 | 0.060 |
| E17 | E19 | logistic | Adding distant tissue | 0.007 | 0.003 |
| E17 | E19 | lda | Adding distant tissue | 0.006 | 0.010 |
| E17 | E19 | svm | Adding distant tissue | 0.004 | 0.009 |
| E17 | E19 | random_forest | Adding distant tissue | -0.001 | 0.005 |
| E17 | E19 | xgboost | Adding distant tissue | 0.002 | 0.001 |
| E17 | E19 | knn | Adding distant tissue | 0.006 | 0.003 |
| E19 | E20 | logistic | Adding controls and global descriptors | 0.008 | 0.035 |
| E19 | E20 | lda | Adding controls and global descriptors | 0.013 | 0.030 |
| E19 | E20 | svm | Adding controls and global descriptors | 0.008 | 0.015 |
| E19 | E20 | random_forest | Adding controls and global descriptors | 0.017 | 0.018 |
| E19 | E20 | xgboost | Adding controls and global descriptors | 0.008 | 0.020 |
| E19 | E20 | knn | Adding controls and global descriptors | 0.013 | 0.014 |


## Group permutation importance on leakage-reduced test

| experiment | model | group | mean | std |
| --- | --- | --- | --- | --- |
| E20 | xgboost | F05 | 0.103 | 0.008 |
| E20 | xgboost | F24 | 0.091 | 0.005 |
| E20 | xgboost | F21 | 0.028 | 0.004 |
| E20 | xgboost | F15 | 0.022 | 0.005 |
| E20 | xgboost | F13 | 0.006 | 0.003 |
| E20 | xgboost | F23 | 0.005 | 0.002 |
| E20 | xgboost | F12 | 0.003 | 0.003 |
| E20 | xgboost | F20 | 0.002 | 0.003 |
| E20 | xgboost | F22 | 0.001 | 0.001 |
| E20 | xgboost | F03 | 0.001 | 0.002 |
| E20 | xgboost | F16 | 0.001 | 0.001 |
| E20 | xgboost | F17 | 0.001 | 0.001 |
| E20 | xgboost | F10 | 0.001 | 0.001 |
| E20 | xgboost | F02 | 0.000 | 0.001 |
| E20 | xgboost | F18 | 0.000 | 0.001 |
| E20 | xgboost | F08 | 0.000 | 0.001 |
| E20 | xgboost | F04 | 0.000 | 0.001 |
| E20 | xgboost | F01 | 0.000 | 0.001 |
| E20 | xgboost | F14 | 0.000 | 0.000 |
| E20 | xgboost | F07 | 0.000 | 0.000 |
| E20 | xgboost | F06 | 0.000 | 0.000 |
| E20 | xgboost | F25 | -0.000 | 0.002 |
| E20 | xgboost | F09 | -0.001 | 0.001 |
| E20 | xgboost | F11 | -0.001 | 0.001 |
| E20 | xgboost | F19 | -0.001 | 0.001 |


See the stage metrics CSVs for both test splits, per-plane scores and group-bootstrap intervals.
High feature-group performance measures predictive information in that group. Use the repository's intervention experiments to study which cues a CNN relies on.
