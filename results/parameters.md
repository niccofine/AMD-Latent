# Parameters
### Parameters employed
In the following, employed parameters across the scripts are summarized and displayed.

* **Classifiers:** Hyperparameters explicitely employed in the three classification tasks are reported in the table below. All the other parameters are set to default.

| Classifier | Hyperparameter | AMD detection | Late-AMD detection | 6-class grading |
| --- | --- | --- | --- | --- |
| Support Vector Machines | Cost parameter $C$ | 1 | 1 | 0.01 |
| Random Forest | Number of trees | 300 | 230 | 170 |
| MultiLayer Perceptron | Number of neurons per layer | (16, 32, 64, 128, 256) | (16, 32) | (16, 32, 64, 128) |
| AdaBoost | Number of estimators | 110 | 240 | 230 |

* **Scaling:** Min-max scaling using `MinMaxScaler` from `scikit-learn` fitted on the construction set, applied along each latent dimension separately.
* **Class weighting:** No class weighting applied.
* **UMAP:** Default `umap-learn` configuration are used to extract UMAPs:
    * `n_neighbors=15`;
    * `n_components=2`;
    * `min_dist=0.1`;
    * `metric='euclidean'`;
    * `spread=1.0`.
* **SHAP:** Based on the different classifier employed, for SHAP analysis the following implementations contained in the `shap` Python library were used:
    * `LinearExplainer` for Linear SVM;
    * `TreeExplainer` for Random Forest.
    
    For each implementation, all the parameters are set as default.
