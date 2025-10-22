import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("xgboost is required for training. Please install it: pip install xgboost") from e

warnings.filterwarnings("ignore")

# --- Configuration constants ---
FEATURES_TO_DROP = ['individual_id', 'address_id']


def parse_args():
    p = argparse.ArgumentParser(description="Train a churn model on a CSV and save artifacts for inference.")
    p.add_argument("--input", required=True, help="Path to input CSV file")
    p.add_argument("--target", default="Churn", help="Target column name (default: Churn)")
    p.add_argument(
        "--drop-cols",
        default=",".join(FEATURES_TO_DROP),
        help="Comma-separated columns to drop before training",
    )
    p.add_argument("--model-out", default="churn_model.pkl", help="Path to save trained model (joblib)")
    p.add_argument("--features-out", default="model_features.pkl", help="Path to save training feature list")
    p.add_argument("--metadata-out", default="model_metadata.json", help="Path to save training metadata JSON")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction for evaluation (default: 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random state for split (default: 42)")
    p.add_argument("--no-refit-full", action="store_true", help="Do not refit on full dataset after evaluation")

    # Search config
    p.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations (default: 20)")
    p.add_argument("--cv-splits", type=int, default=5, help="StratifiedKFold splits (default: 5)")
    p.add_argument("--n-jobs", type=int, default=2, help="Parallel jobs for search (default: 2)")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = args.input
    target_col = args.target
    drop_cols = [c for c in args.drop_cols.split(",") if c]

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data.")

    # Split features/target
    y = df[target_col]
    X_raw = df.drop(columns=[c for c in ([target_col] + drop_cols) if c in df.columns])

    # One-hot encode
    X = pd.get_dummies(X_raw, drop_first=True)
    print(f"Prepared features with shape: {X.shape}")

    # Train/val split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Handle imbalance via scale_pos_weight
    class_counts = y_train.value_counts()
    if 0 in class_counts and 1 in class_counts and class_counts[1] > 0:
        scale_pos_weight = float(class_counts[0]) / float(class_counts[1])
    else:
        scale_pos_weight = 1.0

    # Base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=args.random_state,
        scale_pos_weight=scale_pos_weight,
    )

    # Hyperparameter search space (mirrors model.py)
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.1, 0.5, 1, 1.5, 2],
    }

    kfold = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring='roc_auc',
        cv=kfold,
        verbose=1,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    print("Starting RandomizedSearchCV...")
    search.fit(X_train, y_train)
    print("Best parameters found:")
    print(search.best_params_)

    model = search.best_estimator_

    # Evaluate on holdout using same metrics as model.py
    print("\nEvaluating the final model performance on validation set...")
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_prob)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_pred))

    # Optionally refit on full data for final model
    if not args.no_refit_full:
        print("Refitting best model on full dataset for final artifact...")
        model.fit(X, y)

    # Save artifacts
    print("Saving model and artifacts...")
    joblib.dump(model, args.model_out)
    joblib.dump(list(X.columns), args.features_out)

    metadata = {
        "target": target_col,
        "drop_cols": drop_cols,
        "one_hot": {"drop_first": True},
        "model_path": os.path.abspath(args.model_out),
        "features_path": os.path.abspath(args.features_out),
        "model_type": "xgboost.XGBClassifier",
        # Aid SHAP grouping and regional analysis
        "raw_feature_names": list(X_raw.columns),
        "suggested_group_by": [c for c in [
            "Geographic_Cluster", "state", "city", "county"
        ] if c in df.columns],
        "search": {
            "cv_splits": args.cv_splits,
            "n_iter": args.n_iter,
            "param_grid_keys": list(param_dist.keys()),
        },
    }
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Artifacts saved:")
    print(f"  Model:          {args.model_out}")
    print(f"  Features list:  {args.features_out}")
    print(f"  Metadata JSON:  {args.metadata_out}")
    print("Done.")


if __name__ == "__main__":
    main()
