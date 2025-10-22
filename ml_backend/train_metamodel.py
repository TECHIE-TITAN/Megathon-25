#!/usr/bin/env python3
"""
Logistic Regression Meta-Model
Trains and uses a logistic regression model on outputs from multiple GBM models
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filepath):
    """Load the meta-model dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_features_target(df):
    """Prepare feature matrix and target vector"""
    
    # Target column
    if 'actual_churn' not in df.columns:
        raise ValueError("actual_churn column not found in dataset")
    
    y = df['actual_churn'].astype(int)
    
    # Feature columns (all probability and ensemble features)
    feature_cols = []
    
    # Individual model probabilities
    prob_cols = [col for col in df.columns if col.startswith('prob_') and not col.startswith('prob_mean')]
    feature_cols.extend(prob_cols)
    
    # Ensemble features if available
    ensemble_cols = [col for col in df.columns if col in [
        'prob_mean', 'prob_std', 'prob_min', 'prob_max', 'prob_median', 'prob_range'
    ]]
    feature_cols.extend(ensemble_cols)
    
    # Difference features
    diff_cols = [col for col in df.columns if col.startswith('diff_')]
    feature_cols.extend(diff_cols)
    
    if not feature_cols:
        raise ValueError("No probability features found in dataset")
    
    X = df[feature_cols]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y, feature_cols

def train_metamodel(X, y, feature_names, test_size=0.2, random_state=42):
    """Train logistic regression meta-model"""
    
    print(f"\n{'='*60}")
    print("TRAINING META-MODEL")
    print('='*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression with different regularization strengths
    models = {}
    cv_scores = {}
    
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    for C in C_values:
        model = LogisticRegression(
            C=C, 
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_scores[C] = scores.mean()
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        models[C] = model
        
        print(f"C={C}: CV AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Select best model
    best_C = max(cv_scores, key=cv_scores.get)
    best_model = models[best_C]
    
    print(f"\nBest C: {best_C} (CV AUC: {cv_scores[best_C]:.4f})")
    
    # Evaluate on test set
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_ap = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nTest Set Performance:")
    print(f"AUC: {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': best_model.coef_[0],
        'abs_coefficient': np.abs(best_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeature Importance (top 10):")
    print(feature_importance.head(10))
    
    return {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'feature_importance': feature_importance,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }

def save_model_artifacts(results, output_dir):
    """Save trained model and artifacts"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model and scaler
    joblib.dump(results['model'], output_dir / 'metamodel.pkl')
    joblib.dump(results['scaler'], output_dir / 'metamodel_scaler.pkl')
    
    # Save feature names
    with open(output_dir / 'metamodel_features.txt', 'w') as f:
        for feature in results['feature_names']:
            f.write(f"{feature}\n")
    
    # Save feature importance
    results['feature_importance'].to_csv(output_dir / 'metamodel_feature_importance.csv', index=False)
    
    # Save metadata
    metadata = {
        'test_auc': results['test_auc'],
        'test_ap': results['test_ap'],
        'n_features': len(results['feature_names'])
    }
    
    import json
    with open(output_dir / 'metamodel_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel artifacts saved to: {output_dir}")

def create_visualizations(results, output_dir):
    """Create performance visualizations"""
    
    output_dir = Path(output_dir)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["test_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR Curve (AP = {results["test_ap"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metamodel_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    top_features = results['feature_importance'].head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['abs_coefficient'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Absolute Coefficient')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'metamodel_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def predict_with_metamodel(input_file, model_path, scaler_path, features_path, output_file):
    """Use trained meta-model for predictions"""
    
    print(f"\n{'='*60}")
    print("MAKING PREDICTIONS WITH META-MODEL")
    print('='*60)
    
    # Load model artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded model with {len(feature_names)} features")
    
    # Load input data
    df = pd.read_csv(input_file)
    print(f"Input data: {df.shape}")
    
    # Prepare features
    X = df[feature_names]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Create output dataframe
    output_df = df.copy()
    output_df['final_churn_probability'] = y_pred_proba
    
    # Save results
    output_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to: {output_file}")
    print(f"Final churn probability statistics:")
    print(f"  Mean: {y_pred_proba.mean():.4f}")
    print(f"  Std: {y_pred_proba.std():.4f}")
    print(f"  Min: {y_pred_proba.min():.4f}")
    print(f"  Max: {y_pred_proba.max():.4f}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Train or use logistic regression meta-model")
    parser.add_argument("--mode", choices=['train', 'predict'], required=True, help="Mode: train or predict")
    parser.add_argument("--input", required=True, help="Input dataset file")
    parser.add_argument("--output-dir", default="metamodel_outputs", help="Output directory for trained model")
    parser.add_argument("--output-file", help="Output file for predictions (predict mode)")
    
    # For prediction mode
    parser.add_argument("--model", help="Path to trained model (predict mode)")
    parser.add_argument("--scaler", help="Path to scaler (predict mode)")
    parser.add_argument("--features", help="Path to features file (predict mode)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Training mode
        df = load_dataset(args.input)
        if df is None:
            return False
        
        X, y, feature_names = prepare_features_target(df)
        results = train_metamodel(X, y, feature_names)
        
        # Save artifacts
        save_model_artifacts(results, args.output_dir)
        create_visualizations(results, args.output_dir)
        
        print(f"\nTraining completed successfully!")
        print(f"Use --mode predict to make predictions with the trained model")
        
    elif args.mode == 'predict':
        # Prediction mode
        if not all([args.model, args.scaler, args.features, args.output_file]):
            print("For predict mode, you must specify --model, --scaler, --features, and --output-file")
            return False
        
        predict_with_metamodel(
            args.input, args.model, args.scaler, args.features, args.output_file
        )
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)