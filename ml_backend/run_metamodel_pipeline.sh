#!/bin/bash
# Complete Meta-Model Pipeline Example
# This script demonstrates the full pipeline from multi-model inference to final predictions

echo "=================================================="
echo "META-MODEL PIPELINE EXAMPLE"
echo "=================================================="

# Step 1: Run inference across all models
echo ""
echo "STEP 1: Running inference across all GBM models..."
echo "Command: python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs"
echo ""

# Uncomment to run:
# python run_multi_inference.py --input for_inf_after37.5csv --output-dir multi_model_outputs

# Step 2: Prepare meta-model dataset
echo ""
echo "STEP 2: Preparing meta-model dataset..."
echo "Command: python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble"
echo ""

# Uncomment to run:
# python prepare_metamodel_dataset.py --input-dir multi_model_outputs --output metamodel_dataset.csv --add-ensemble

# Step 3: Train meta-model
echo ""
echo "STEP 3: Training logistic regression meta-model..."
echo "Command: python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts"
echo ""

# Uncomment to run:
# python train_metamodel.py --mode train --input metamodel_dataset.csv --output-dir metamodel_artifacts

# Step 4: Make final predictions
echo ""
echo "STEP 4: Making final predictions with meta-model..."
echo "Command: python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv"
echo ""

# Uncomment to run:
# python train_metamodel.py --mode predict --input metamodel_dataset.csv --model metamodel_artifacts/metamodel.pkl --scaler metamodel_artifacts/metamodel_scaler.pkl --features metamodel_artifacts/metamodel_features.txt --output-file final_churn_predictions.csv

echo ""
echo "=================================================="
echo "PIPELINE OVERVIEW"
echo "=================================================="
echo "1. run_multi_inference.py - Runs x.py for all 4 models"
echo "2. prepare_metamodel_dataset.py - Combines outputs into training dataset"
echo "3. train_metamodel.py (train mode) - Trains logistic regression meta-model"
echo "4. train_metamodel.py (predict mode) - Makes final predictions"
echo ""
echo "Expected files after completion:"
echo "  multi_model_outputs/inference_37.5k_v2.csv"
echo "  multi_model_outputs/inference_37.5k_v3.csv" 
echo "  multi_model_outputs/inference_37.5k.csv"
echo "  multi_model_outputs/inference_full.csv"
echo "  metamodel_dataset.csv"
echo "  metamodel_artifacts/metamodel.pkl"
echo "  metamodel_artifacts/metamodel_scaler.pkl"
echo "  metamodel_artifacts/metamodel_features.txt"
echo "  final_churn_predictions.csv"
echo ""
echo "To run the full pipeline, uncomment the python commands in this script."