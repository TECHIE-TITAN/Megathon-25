import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
import joblib # Added for saving the model

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
CSV_FILE_PATH = 'csv_files/autoinsurance_churn_50000.csv'
TARGET_COLUMN = 'Churn'
FEATURES_TO_DROP = ['individual_id', 'address_id']

# --- 2. Load and Prepare the Data ---
print("Loading and preparing data...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    df = pd.DataFrame({
        'Age': np.random.randint(20, 70, 100), 'Income': np.random.randint(30000, 150000, 100),
        'CreditScore': np.random.randint(300, 850, 100), 'Months_on_Book': np.random.randint(1, 60, 100),
        'clustering_label_1': np.random.randint(0, 4, 100), 'Gender': ['M', 'F'] * 50,
        TARGET_COLUMN: np.random.randint(0, 2, 100)
    })
    print("A dummy dataframe has been created to allow the script to run.")

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN] + FEATURES_TO_DROP)
X = pd.get_dummies(X, drop_first=True)
###
# Persist the exact training feature list early so inference can align columns without requiring a full retrain
#try:
 #   joblib.dump(list(X.columns), 'model_features.pkl')
 #   print(f"Saved training feature list early to 'model_features.pkl' ({len(X.columns)} features).")
#except Exception as e:
  #  print(f"Warning: Couldn't save model_features.pkl early: {e}")
#*/
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Data prepared. Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
print("-" * 30)


# --- 3. Hyperparameter Tuning with RandomizedSearchCV ---
print("Starting hyperparameter tuning with RandomizedSearchCV...")

# Define the expanded hyperparameter grid to search
param_dist = {
    # Number of trees in the forest
    'n_estimators': [100, 200, 300, 400],
    
    # Step size shrinkage to prevent overfitting
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    
    # Maximum depth of a tree
    'max_depth': [3, 4, 5, 6],
    
    # Fraction of samples to be used for fitting each tree
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Fraction of features to be used for fitting each tree
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Minimum loss reduction required to make a further partition
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    
    # L1 regularization term on weights (helps prevent overfitting)
    'reg_alpha': [0, 0.1, 0.5, 1],

    # L2 regularization term on weights (helps prevent overfitting)
    'reg_lambda': [0.1, 0.5, 1, 1.5, 2]
}

# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Use Stratified K-Fold for cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,  # Increase this for a more thorough search
    scoring='roc_auc',
    cv=kfold,
    verbose=1,
    random_state=42,
    n_jobs=2
)

# Fit the random search model
random_search.fit(X_train, y_train)

# Get the best model and its parameters
print("\nBest parameters found:")
print(random_search.best_params_)

# The best model is now stored in 'best_estimator_'
model = random_search.best_estimator_

print("\nModel training with best parameters complete.")
print("-" * 30)


# --- 4. Evaluate the Best Model ---
print("Evaluating the final model performance...")
preds = model.predict(X_test)
pred_probs = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, pred_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("-" * 30)


# --- 5. Integrate SHAP for Explainability ---
print("Setting up SHAP for explainability...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP setup complete. You can now generate plots.")
print("-" * 30)


# --- Example SHAP Visualizations ---
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

print("Generating SHAP force plot for a single customer...")
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False)


# --- 6. Save Model and Explainer ---
print("\nSaving the final model and SHAP explainer...")

# Save the trained model to a file named 'churn_model.pkl'
joblib.dump(model, 'churn_model_50k_no_cluster.pkl')

# Save the SHAP explainer to a file named 'shap_explainer.pkl'
joblib.dump(explainer, 'shap_explainer_50k_no_cluster.pkl')

# Also save the exact training feature list for inference alignment
joblib.dump(list(X.columns), 'model_features_50k_no_cluster.pkl')
print(f"Saved training feature list to 'model_features.pkl' ({len(X.columns)} features).")

print("Model and explainer saved successfully.")
print("Script finished.")