#!/usr/bin/env python3
"""
Meta-Model Dataset Preparation
Combines outputs from multiple GBM models to create dataset for logistic regression meta-model
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

def load_model_output(filepath, model_name):
    """Load a model output CSV and extract relevant columns"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {model_name}: {df.shape}")
        
        # Get column names
        cols = df.columns.tolist()
        print(f"  Columns: {cols}")
        
        # The last column should be churn probability
        prob_col = cols[-1]
        
        # Look for actual churn column - prioritize 'Churn' column, then 7th column (index 6)
        actual_churn_col = None
        if 'Churn' in cols:
            actual_churn_col = 'Churn'
        elif len(cols) >= 7:
            actual_churn_col = cols[6]  # 7th column from left (index 6)
        else:
            print(f"  Warning: {model_name} has only {len(cols)} columns, cannot find actual churn")
            actual_churn_col = None
        
        # Extract individual_id if present (usually first column)
        individual_id_col = None
        if 'individual_id' in cols:
            individual_id_col = 'individual_id'
        elif cols[0].lower().startswith('individual'):
            individual_id_col = cols[0]
        
        result = {
            'model_name': model_name,
            'dataframe': df,
            'prob_column': prob_col,
            'actual_churn_column': actual_churn_col,
            'individual_id_column': individual_id_col,
            'n_rows': len(df)
        }
        
        print(f"  Probability column: {prob_col}")
        print(f"  Actual churn column: {actual_churn_col}")
        print(f"  Individual ID column: {individual_id_col}")
        
        return result
        
    except Exception as e:
        print(f"Error loading {model_name} from {filepath}: {e}")
        return None

def align_datasets(model_outputs):
    """Align datasets on individual_id or row order"""
    
    # Check if all models have individual_id
    has_id = [output for output in model_outputs if output['individual_id_column'] is not None]
    
    if len(has_id) == len(model_outputs):
        print("All models have individual_id, aligning by ID...")
        return align_by_id(model_outputs)
    else:
        print("Not all models have individual_id, aligning by row order...")
        return align_by_order(model_outputs)

def align_by_id(model_outputs):
    """Align datasets using individual_id"""
    
    # Get the first dataset as base
    base_output = model_outputs[0]
    base_df = base_output['dataframe']
    id_col = base_output['individual_id_column']
    
    # Start with individual_id and actual churn from base
    result_df = base_df[[id_col]].copy()
    
    # Add actual churn from first model (should be same across all)
    if base_output['actual_churn_column']:
        result_df['actual_churn'] = base_df[base_output['actual_churn_column']]
    
    # Add probabilities from each model
    for output in model_outputs:
        df = output['dataframe']
        model_name = output['model_name']
        prob_col = output['prob_column']
        model_id_col = output['individual_id_column']
        
        # Merge on individual_id
        prob_data = df[[model_id_col, prob_col]].rename(columns={
            prob_col: f'prob_{model_name}'
        })
        
        result_df = result_df.merge(
            prob_data, 
            left_on=id_col, 
            right_on=model_id_col, 
            how='inner'
        )
        
        # Drop the duplicate id column from merge
        if model_id_col != id_col:
            result_df = result_df.drop(columns=[model_id_col])
        
        print(f"After merging {model_name}: {result_df.shape}")
    
    return result_df

def align_by_order(model_outputs):
    """Align datasets by row order (assuming same order)"""
    
    # Find minimum number of rows
    min_rows = min(output['n_rows'] for output in model_outputs)
    print(f"Aligning by row order, using first {min_rows} rows from each model")
    
    # Start with row indices
    result_df = pd.DataFrame({'row_index': range(min_rows)})
    
    # Add individual_id if available from any model
    for output in model_outputs:
        if output['individual_id_column']:
            df = output['dataframe'].head(min_rows)
            result_df['individual_id'] = df[output['individual_id_column']].values
            break
    
    # Add actual churn from first model
    first_output = model_outputs[0]
    if first_output['actual_churn_column']:
        first_df = first_output['dataframe'].head(min_rows)
        result_df['actual_churn'] = first_df[first_output['actual_churn_column']].values
    
    # Add probabilities from each model
    for output in model_outputs:
        df = output['dataframe'].head(min_rows)
        model_name = output['model_name']
        prob_col = output['prob_column']
        
        result_df[f'prob_{model_name}'] = df[prob_col].values
        print(f"Added {model_name} probabilities: {result_df.shape}")
    
    return result_df

def add_ensemble_features(df):
    """Add ensemble features like mean, std, min, max of probabilities"""
    
    # Get probability columns
    prob_cols = [col for col in df.columns if col.startswith('prob_')]
    
    if len(prob_cols) < 2:
        print("Warning: Less than 2 probability columns, skipping ensemble features")
        return df
    
    print(f"Adding ensemble features from {len(prob_cols)} probability columns")
    
    # Calculate ensemble statistics
    prob_data = df[prob_cols]
    
    df['prob_mean'] = prob_data.mean(axis=1)
    df['prob_std'] = prob_data.std(axis=1)
    df['prob_min'] = prob_data.min(axis=1)
    df['prob_max'] = prob_data.max(axis=1)
    df['prob_median'] = prob_data.median(axis=1)
    
    # Agreement features
    df['prob_range'] = df['prob_max'] - df['prob_min']
    
    # Pairwise differences for model agreement
    if len(prob_cols) >= 2:
        for i in range(len(prob_cols)):
            for j in range(i+1, len(prob_cols)):
                col1, col2 = prob_cols[i], prob_cols[j]
                diff_name = f'diff_{col1.replace("prob_", "")}_{col2.replace("prob_", "")}'
                df[diff_name] = abs(df[col1] - df[col2])
    
    print(f"Final dataset shape: {df.shape}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Prepare meta-model dataset from multiple GBM outputs")
    parser.add_argument("--input-dir", default="multi_model_outputs", help="Directory with model outputs")
    parser.add_argument("--output", default="metamodel_dataset.csv", help="Output dataset file")
    parser.add_argument("--add-ensemble", action="store_true", help="Add ensemble features")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
    # Find all CSV files in input directory
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {input_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")
    
    # Load all model outputs
    model_outputs = []
    for csv_file in csv_files:
        model_name = csv_file.stem.replace("inference_", "")
        output = load_model_output(csv_file, model_name)
        if output:
            model_outputs.append(output)
    
    if not model_outputs:
        print("Error: No valid model outputs loaded")
        return False
    
    print(f"\nSuccessfully loaded {len(model_outputs)} model outputs")
    
    # Align datasets
    print(f"\n{'='*60}")
    print("ALIGNING DATASETS")
    print('='*60)
    
    aligned_df = align_datasets(model_outputs)
    
    print(f"Aligned dataset shape: {aligned_df.shape}")
    print(f"Columns: {aligned_df.columns.tolist()}")
    
    # Add ensemble features if requested
    if args.add_ensemble:
        print(f"\n{'='*60}")
        print("ADDING ENSEMBLE FEATURES")
        print('='*60)
        aligned_df = add_ensemble_features(aligned_df)
    
    # Save dataset
    aligned_df.to_csv(args.output, index=False)
    print(f"\nDataset saved to: {args.output}")
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print('='*60)
    print(f"Total rows: {len(aligned_df)}")
    print(f"Total columns: {len(aligned_df.columns)}")
    
    # Show actual churn distribution if available
    if 'actual_churn' in aligned_df.columns:
        churn_dist = aligned_df['actual_churn'].value_counts()
        print(f"\nActual churn distribution:")
        for value, count in churn_dist.items():
            pct = count / len(aligned_df) * 100
            print(f"  {value}: {count} ({pct:.1f}%)")
    
    # Show probability column statistics
    prob_cols = [col for col in aligned_df.columns if col.startswith('prob_')]
    if prob_cols:
        print(f"\nProbability columns statistics:")
        print(aligned_df[prob_cols].describe())
    
    print(f"\nNext step: Use train_metamodel.py with {args.output}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)