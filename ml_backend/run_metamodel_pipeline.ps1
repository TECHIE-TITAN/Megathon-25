# Complete Meta-Model Pipeline Example (PowerShell)
# This script demonstrates the full pipeline from multi-model inference to final predictions

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "META-MODEL PIPELINE EXAMPLE" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Run inference across all models
Write-Host ""
Write-Host "STEP 1: Running inference across all GBM models..." -ForegroundColor Yellow
Write-Host "Command: python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs" -ForegroundColor Green
Write-Host ""

# Uncomment to run:
# python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs

# Step 2: Prepare meta-model dataset
Write-Host ""
Write-Host "STEP 2: Preparing meta-model dataset..." -ForegroundColor Yellow
Write-Host "Command: python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble" -ForegroundColor Green
Write-Host ""

# Uncomment to run:
# python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble

# Step 3: Train meta-model
Write-Host ""
Write-Host "STEP 3: Training logistic regression meta-model..." -ForegroundColor Yellow
Write-Host "Command: python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts" -ForegroundColor Green
Write-Host ""

# Uncomment to run:
# python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts

# Step 4: Make final predictions
Write-Host ""
Write-Host "STEP 4: Making final predictions with meta-model..." -ForegroundColor Yellow
Write-Host "Command: python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv" -ForegroundColor Green
Write-Host ""

# Uncomment to run:
# python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "PIPELINE OVERVIEW" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "1. run_multi_inference.py - Runs x.py for all 4 models" -ForegroundColor White
Write-Host "2. prepare_metamodel_dataset.py - Combines outputs into training dataset" -ForegroundColor White
Write-Host "3. train_metamodel.py (train mode) - Trains logistic regression meta-model" -ForegroundColor White
Write-Host "4. train_metamodel.py (predict mode) - Makes final predictions" -ForegroundColor White
Write-Host ""
Write-Host "Expected files after completion:" -ForegroundColor Magenta
Write-Host "  multi_model_outputs/inference_37.5k_v2.csv" -ForegroundColor Gray
Write-Host "  multi_model_outputs/inference_37.5k_v3.csv" -ForegroundColor Gray
Write-Host "  multi_model_outputs/inference_37.5k.csv" -ForegroundColor Gray
Write-Host "  multi_model_outputs/inference_full.csv" -ForegroundColor Gray
Write-Host "  metamodel_dataset.csv" -ForegroundColor Gray
Write-Host "  metamodel_artifacts/metamodel.pkl" -ForegroundColor Gray
Write-Host "  metamodel_artifacts/metamodel_scaler.pkl" -ForegroundColor Gray
Write-Host "  metamodel_artifacts/metamodel_features.txt" -ForegroundColor Gray
Write-Host "  final_churn_predictions.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "To run the full pipeline, uncomment the python commands in this script." -ForegroundColor Red