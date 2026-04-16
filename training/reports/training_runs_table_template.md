| Candidate | MLflow run link | Code version (git sha) | Key hyperparams | Key model metrics | Key training cost metrics | Notes |
|---|---|---|---|---|---|---|
| baseline_tfidf_logreg |  |  |  | val_macro_f1=, test_macro_f1=, test_accuracy= | train_seconds=, feature_build_seconds=, model_artifact_size_bytes= | Fast CPU baseline |
| tfidf_lightgbm |  |  |  | val_macro_f1=, test_macro_f1=, test_accuracy= | train_seconds=, feature_build_seconds=, model_artifact_size_bytes= | Tradeoff candidate |
| sbert_logreg |  |  |  | val_macro_f1=, test_macro_f1=, test_accuracy= | train_seconds=, feature_build_seconds=, model_artifact_size_bytes= | Better semantics |
| sbert_mlp |  |  |  | val_macro_f1=, test_macro_f1=, test_accuracy= | train_seconds=, feature_build_seconds=, model_artifact_size_bytes= | Highest capacity |
