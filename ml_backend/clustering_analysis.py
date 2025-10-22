import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChurnKClusteringAnalysis:
    def __init__(self, data_path, n_samples=37499):
        """
        Initialize the K-Clustering Analysis for Churn Prediction
        
        Args:
            data_path (str): Path to the autoinsurance_churn.csv file
            n_samples (int): Number of samples to use (default: 37499)
        """
        self.data_path = data_path
        self.n_samples = n_samples
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = None
        self.cluster_features = None
        
    def load_and_sample_data(self):
        """Load the first n_samples from the dataset"""
        print(f"Loading first {self.n_samples} samples from {self.data_path}...")
        
        # Read only the first n_samples + 1 (including header)
        self.data = pd.read_csv(self.data_path, nrows=self.n_samples)
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Display basic info
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nChurn distribution:")
        print(self.data['Churn'].value_counts())
        print(f"Churn rate: {self.data['Churn'].mean():.4f}")
        
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for clustering"""
        print("\nPreprocessing data for clustering...")
        
        # Create a copy for processing
        df = self.data.copy()
        
        # Handle missing values
        print("Handling missing values...")
        missing_summary = df.isnull().sum()
        print(f"Missing values per column:\n{missing_summary[missing_summary > 0]}")
        
        # Drop columns with too many missing values or not useful for clustering
        cols_to_drop = ['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Handle categorical variables
        categorical_cols = ['city', 'state', 'county', 'marital_status', 'home_market_value']
        
        for col in categorical_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Fill missing values with 'Unknown'
                    df[col] = df[col].fillna('Unknown')
                    
                    # Label encode categorical variables
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
        
        # Handle numerical variables
        numerical_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 'longitude', 
                         'income', 'has_children', 'length_of_residence', 'home_owner', 
                         'college_degree', 'good_credit']
        
        for col in numerical_cols:
            if col in df.columns:
                # Fill missing values with median
                df[col] = df[col].fillna(df[col].median())
        
        # Separate features and target
        target = 'Churn'
        feature_cols = [col for col in df.columns if col != target]
        
        X = df[feature_cols]
        y = df[target]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store processed data
        self.processed_data = pd.DataFrame(X_scaled, columns=feature_cols)
        self.processed_data['Churn'] = y.reset_index(drop=True)
        self.cluster_features = feature_cols
        
        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Features used for clustering: {len(feature_cols)}")
        
        return self.processed_data
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print(f"\nFinding optimal number of clusters (k=2 to {max_k})...")
        
        X = self.processed_data[self.cluster_features]
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal k based on silhouette score: {optimal_k}")
        
        return optimal_k, silhouette_scores, inertias
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_clusters()
        
        print(f"\nPerforming K-means clustering with k={n_clusters}...")
        
        X = self.processed_data[self.cluster_features]
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X)
        
        # Add cluster labels to the data
        self.processed_data['Cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        # Analyze clusters
        self.analyze_clusters()
        
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze the characteristics of each cluster"""
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS")
        print("="*80)
        
        cluster_summary = []
        
        for cluster_id in sorted(self.processed_data['Cluster'].unique()):
            cluster_data = self.processed_data[self.processed_data['Cluster'] == cluster_id]
            
            summary = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(self.processed_data) * 100,
                'Churn_Rate': cluster_data['Churn'].mean(),
                'Churn_Count': cluster_data['Churn'].sum()
            }
            
            # Add feature means (original scale)
            original_data = self.data.iloc[cluster_data.index]
            for col in ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income']:
                if col in original_data.columns:
                    summary[f'Avg_{col}'] = original_data[col].mean()
            
            cluster_summary.append(summary)
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {summary['Size']} ({summary['Percentage']:.1f}%)")
            print(f"  Churn Rate: {summary['Churn_Rate']:.4f}")
            print(f"  Churn Count: {summary['Churn_Count']}")
            
            if 'Avg_curr_ann_amt' in summary:
                print(f"  Avg Annual Amount: ${summary['Avg_curr_ann_amt']:.2f}")
            if 'Avg_days_tenure' in summary:
                print(f"  Avg Days Tenure: {summary['Avg_days_tenure']:.0f}")
            if 'Avg_age_in_years' in summary:
                print(f"  Avg Age: {summary['Avg_age_in_years']:.1f}")
            if 'Avg_income' in summary:
                print(f"  Avg Income: ${summary['Avg_income']:.2f}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(cluster_summary)
        
        # Visualize cluster characteristics
        self.visualize_clusters(summary_df)
        
        return summary_df
    
    def visualize_clusters(self, summary_df):
        """Create visualizations for cluster analysis"""
        plt.figure(figsize=(20, 15))
        
        # 1. Cluster sizes
        plt.subplot(3, 3, 1)
        plt.pie(summary_df['Size'], labels=[f'Cluster {i}' for i in summary_df['Cluster']], autopct='%1.1f%%')
        plt.title('Cluster Size Distribution')
        
        # 2. Churn rate by cluster
        plt.subplot(3, 3, 2)
        bars = plt.bar(summary_df['Cluster'], summary_df['Churn_Rate'])
        plt.title('Churn Rate by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Churn Rate')
        plt.ylim(0, 1)
        for bar, rate in zip(bars, summary_df['Churn_Rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 3. Cluster vs Churn heatmap
        plt.subplot(3, 3, 3)
        cluster_churn = pd.crosstab(self.processed_data['Cluster'], self.processed_data['Churn'], normalize='index')
        sns.heatmap(cluster_churn, annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title('Cluster vs Churn Distribution')
        
        # 4-9. Feature distributions by cluster (if available)
        feature_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'latitude', 'longitude']
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        for i, col in enumerate(available_features[:6]):
            plt.subplot(3, 3, 4 + i)
            for cluster_id in sorted(self.processed_data['Cluster'].unique()):
                cluster_indices = self.processed_data[self.processed_data['Cluster'] == cluster_id].index
                cluster_values = self.data.iloc[cluster_indices][col].dropna()
                plt.hist(cluster_values, alpha=0.6, label=f'Cluster {cluster_id}', bins=20)
            plt.title(f'{col} Distribution by Cluster')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_gbm_dataset(self):
        """Prepare the final dataset for GBM model training"""
        print("\n" + "="*80)
        print("PREPARING DATASET FOR GBM MODEL")
        print("="*80)
        
        # Create feature engineering based on clusters
        gbm_data = self.data.copy()
        
        # Add cluster information
        gbm_data['Cluster'] = self.processed_data['Cluster']
        
        # Create cluster-based features
        cluster_stats = self.processed_data.groupby('Cluster')['Churn'].agg(['mean', 'count']).reset_index()
        cluster_stats.columns = ['Cluster', 'Cluster_Churn_Rate', 'Cluster_Size']
        
        gbm_data = gbm_data.merge(cluster_stats, on='Cluster', how='left')
        
        # Feature engineering
        print("Creating additional features...")
        
        # Age groups
        gbm_data['Age_Group'] = pd.cut(gbm_data['age_in_years'], 
                                      bins=[0, 30, 45, 60, 100], 
                                      labels=['Young', 'Middle', 'Senior', 'Elder'])
        
        # Income groups
        gbm_data['Income_Group'] = pd.cut(gbm_data['income'], 
                                         bins=[0, 30000, 60000, 100000, float('inf')], 
                                         labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Tenure groups
        gbm_data['Tenure_Group'] = pd.cut(gbm_data['days_tenure'], 
                                         bins=[0, 365, 1095, 1825, float('inf')], 
                                         labels=['New', 'Medium', 'Long', 'Very_Long'])
        
        # High risk indicators
        gbm_data['High_Risk_Cluster'] = (gbm_data['Cluster_Churn_Rate'] > gbm_data['Cluster_Churn_Rate'].median()).astype(int)
        
        # Handle missing values and encode categoricals for GBM
        print("Encoding categorical variables...")
        categorical_columns = ['city', 'state', 'county', 'marital_status', 'home_market_value', 
                              'Age_Group', 'Income_Group', 'Tenure_Group']
        
        for col in categorical_columns:
            if col in gbm_data.columns:
                gbm_data[col] = gbm_data[col].astype(str).fillna('Unknown')
                le = LabelEncoder()
                gbm_data[col + '_encoded'] = le.fit_transform(gbm_data[col])
        
        # Fill numerical missing values
        numerical_columns = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 'longitude', 
                           'income', 'has_children', 'length_of_residence', 'home_owner', 
                           'college_degree', 'good_credit']
        
        for col in numerical_columns:
            if col in gbm_data.columns:
                gbm_data[col] = gbm_data[col].fillna(gbm_data[col].median())
        
        # Select final features for GBM
        feature_columns = []
        
        # Numerical features
        feature_columns.extend([col for col in numerical_columns if col in gbm_data.columns])
        
        # Encoded categorical features
        feature_columns.extend([col for col in gbm_data.columns if col.endswith('_encoded')])
        
        # Cluster features
        feature_columns.extend(['Cluster', 'Cluster_Churn_Rate', 'Cluster_Size', 'High_Risk_Cluster'])
        
        # Final dataset
        final_features = feature_columns + ['Churn']
        gbm_dataset = gbm_data[final_features].copy()
        
        print(f"Final GBM dataset shape: {gbm_dataset.shape}")
        print(f"Features for GBM: {len(feature_columns)}")
        print(f"Feature list: {feature_columns}")
        
        # Save the dataset
        output_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/gbm_training_dataset.csv'
        gbm_dataset.to_csv(output_path, index=False)
        print(f"\nGBM training dataset saved to: {output_path}")
        
        # Create train/test split
        X = gbm_dataset.drop('Churn', axis=1)
        y = gbm_dataset['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save train/test splits
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/gbm_train_data.csv'
        test_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/gbm_test_data.csv'
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"Training data saved to: {train_path} (shape: {train_data.shape})")
        print(f"Test data saved to: {test_path} (shape: {test_data.shape})")
        
        # Create feature importance summary
        feature_summary = pd.DataFrame({
            'Feature': feature_columns,
            'Type': ['Numerical' if not col.endswith('_encoded') and col not in ['Cluster', 'Cluster_Churn_Rate', 'Cluster_Size', 'High_Risk_Cluster'] 
                    else 'Categorical_Encoded' if col.endswith('_encoded')
                    else 'Cluster_Feature' for col in feature_columns]
        })
        
        feature_summary_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/feature_summary.csv'
        feature_summary.to_csv(feature_summary_path, index=False)
        print(f"Feature summary saved to: {feature_summary_path}")
        
        return gbm_dataset, feature_columns
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/k-clustering/k_clustering_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("K-CLUSTERING ANALYSIS REPORT FOR CHURN PREDICTION\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Samples analyzed: {self.n_samples}\n")
            f.write(f"Original data shape: {self.data.shape}\n")
            f.write(f"Processed data shape: {self.processed_data.shape}\n\n")
            
            f.write("CLUSTER INFORMATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of clusters: {len(self.processed_data['Cluster'].unique())}\n")
            f.write(f"Clustering algorithm: K-Means\n")
            f.write(f"Features used: {len(self.cluster_features)}\n\n")
            
            # Cluster summary
            cluster_summary = self.processed_data.groupby('Cluster').agg({
                'Churn': ['count', 'sum', 'mean']
            }).round(4)
            
            f.write("CLUSTER SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(cluster_summary.to_string())
            f.write("\n\n")
            
            f.write("FEATURES FOR GBM MODEL:\n")
            f.write("-" * 50 + "\n")
            for i, feature in enumerate(self.cluster_features, 1):
                f.write(f"{i:2d}. {feature}\n")
            
            f.write("\nADDITIONAL CLUSTER-BASED FEATURES CREATED:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Cluster (cluster assignment)\n")
            f.write("2. Cluster_Churn_Rate (churn rate of the cluster)\n")
            f.write("3. Cluster_Size (size of the cluster)\n")
            f.write("4. High_Risk_Cluster (binary indicator for high churn clusters)\n")
            f.write("5. Age_Group (categorical age groups)\n")
            f.write("6. Income_Group (categorical income groups)\n")
            f.write("7. Tenure_Group (categorical tenure groups)\n\n")
            
            f.write("OUTPUT FILES CREATED:\n")
            f.write("-" * 50 + "\n")
            f.write("1. gbm_training_dataset.csv - Complete dataset for GBM training\n")
            f.write("2. gbm_train_data.csv - Training split (80%)\n")
            f.write("3. gbm_test_data.csv - Test split (20%)\n")
            f.write("4. feature_summary.csv - Feature information\n")
            f.write("5. optimal_clusters.png - Cluster optimization plots\n")
            f.write("6. cluster_analysis.png - Cluster analysis visualizations\n")
            f.write("7. k_clustering_analysis_report.txt - This report\n\n")
            
            f.write("NEXT STEPS FOR GBM MODEL:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Load gbm_train_data.csv for training\n")
            f.write("2. Use gbm_test_data.csv for evaluation\n")
            f.write("3. Consider the cluster-based features for improved predictions\n")
            f.write("4. Pay attention to high-risk clusters identified\n")
            f.write("5. Use feature_summary.csv to understand feature types\n")
        
        print(f"Comprehensive report saved to: {report_path}")

def main():
    """Main execution function"""
    print("="*100)
    print("K-CLUSTERING ANALYSIS FOR CHURN PREDICTION")
    print("="*100)
    
    # Initialize the analysis
    data_path = '/home/vansh-goyal/Desktop/WorkSpace/Hackathons/ECell Megathon 2025/Data Preprocessing/autoinsurance_churn.csv'
    analyzer = ChurnKClusteringAnalysis(data_path, n_samples=37499)
    
    try:
        # Step 1: Load and sample data
        analyzer.load_and_sample_data()
        
        # Step 2: Preprocess data
        analyzer.preprocess_data()
        
        # Step 3: Find optimal clusters and perform clustering
        analyzer.perform_clustering()
        
        # Step 4: Prepare GBM dataset
        analyzer.prepare_gbm_dataset()
        
        # Step 5: Generate comprehensive report
        analyzer.generate_report()
        
        print("\n" + "="*100)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*100)
        print("Check the k-clustering folder for all generated files:")
        print("- GBM training datasets")
        print("- Visualizations")
        print("- Analysis report")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()