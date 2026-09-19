# Model training

The expected training workflow is:

1. Collect labeled molecular data.
2. Validate and standardize molecular structures.
3. Generate descriptors or fingerprints.
4. Split the data into training, validation, and test sets.
5. Train candidate regression or classification models.
6. Tune hyperparameters.
7. Evaluate on held-out data.
8. Save the model and preprocessing pipeline.

Recommended metrics include MAE, RMSE, R², Pearson correlation, and Spearman correlation for regression tasks. The dataset version, feature-generation method, hyperparameters, and evaluation metrics should be recorded for reproducibility.
