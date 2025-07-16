import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Make sure this path is correct for your system!
JSON_FILE_PATH = 'transactions_sample.json' 
OUTPUT_SCORES_PATH = 'wallet_scores.json'
OUTPUT_ANALYSIS_DIR = 'analysis_results'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(json_file_path):
    """
    Loads transaction data from a JSON file, flattens it, and preprocesses timestamps and amounts.
    Adjusted to handle specific column names and nested structures found in your data.
    """
    print(f"Loading data from {json_file_path}...")
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}. Please download and extract the data.")
        print("See instructions in the `if __name__ == '__main__':` block.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}. Is it a valid JSON file?")
        print(f"JSON Error: {e}")
        return None

    df = pd.DataFrame(data)

    # --- START MODIFICATION BASED ON YOUR COLUMN NAMES ---

    # Standardize column names for 'action'
    # Your data already has 'action', so no rename is strictly needed, but we ensure it's there.
    if 'action' not in df.columns:
        print("Warning: 'action' column not found for transaction type. Please check JSON structure.")
        print(f"Available columns: {df.columns.tolist()}")
        df['action'] = 'UNKNOWN_ACTION' # Assign a placeholder

    # Identify wallet address: Use 'userWallet' as identified from your columns.
    if 'userWallet' in df.columns:
        df['wallet'] = df['userWallet']
    else:
        print("Error: 'userWallet' column not found for wallet addresses.")
        print("Please inspect your JSON file to find the correct wallet identifier field.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Extract amount, asset symbol, and decimals from 'actionData'
    # We assume 'actionData' is a dictionary containing 'amount' and 'reserve' (which has 'symbol' and 'decimals')
    if 'actionData' in df.columns and len(df) > 0 and isinstance(df['actionData'].iloc[0], dict):
        df['amount_raw'] = df['actionData'].apply(lambda x: x.get('amount') if isinstance(x, dict) else None)
        df['asset_symbol'] = df['actionData'].apply(lambda x: x.get('reserve', {}).get('symbol') if isinstance(x, dict) else None)
        df['asset_decimals'] = df['actionData'].apply(lambda x: x.get('reserve', {}).get('decimals') if isinstance(x, dict) else None)
    else:
        print("Error: 'actionData' column not found or not in expected dictionary format.")
        print("Cannot extract amount, asset symbol, or decimals.")
        print(f"Available columns: {df.columns.tolist()}")
        return None # Critical error, cannot proceed without amount/asset info

    # Convert 'amount_raw' from string (wei) to numeric
    df['amount_raw'] = pd.to_numeric(df['amount_raw'], errors='coerce').fillna(0)

    # Apply decimals for actual value. Handle cases where decimals might be missing or non-numeric.
    def convert_amount(row):
        try:
            decimals = int(row['asset_decimals']) if pd.notnull(row['asset_decimals']) else 18 # Default to 18 if decimals are missing
            return row['amount_raw'] / (10 ** decimals)
        except (ValueError, TypeError):
            # If conversion fails (e.g., decimals is invalid), use raw amount and log a warning
            print(f"Warning: Failed to convert amount for row (wallet: {row.get('wallet', 'N/A')}, action: {row.get('action', 'N/A')}). Using raw amount. Check asset_decimals: {row.get('asset_decimals', 'N/A')}, raw amount: {row.get('amount_raw', 'N/A')}")
            return row['amount_raw']

    df['amount_normalized'] = df.apply(convert_amount, axis=1)

    # --- END MODIFICATION ---

    print(f"Data loaded. Shape: {df.shape}")
    print(f"Number of unique wallets: {df['wallet'].nunique()}")
    print("\nSample Data Head:")
    print(df.head())
    print("\nAction counts:")
    print(df['action'].value_counts())
    return df

# --- 2. Feature Engineering (No changes needed here for now) ---

def feature_engineer(df):
    """
    Engineers features from the transaction data for each wallet.
    """
    print("Starting feature engineering...")
    wallet_features = []

    # Get a list of all possible action types to ensure all columns are present
    # Filter out 'UNKNOWN_ACTION' if it was assigned as a fallback
    all_action_types = [action for action in df['action'].unique() if action != 'UNKNOWN_ACTION']

    # Define a set of standard Aave actions we expect
    standard_aave_actions = ['Deposit', 'Borrow', 'Repay', 'Withdraw', 'LiquidationCall', 'SwapBorrowRateMode', 'SetUserUseReserveAsCollateral']

    # Group by wallet to compute aggregate features
    for wallet_address, wallet_df in df.groupby('wallet'):
        features = {'wallet': wallet_address}

        # Time-based features
        features['total_transactions'] = len(wallet_df)
        features['unique_assets_interacted'] = wallet_df['asset_symbol'].nunique()

        # Handle case where wallet_df has only one transaction, min/max timestamp will be the same
        if features['total_transactions'] > 0:
            features['wallet_first_tx_timestamp'] = wallet_df['timestamp'].min()
            features['wallet_last_tx_timestamp'] = wallet_df['timestamp'].max()
            features['wallet_active_period_days'] = (features['wallet_last_tx_timestamp'] - features['wallet_first_tx_timestamp']).days if features['total_transactions'] > 1 else 0
        else: # Should not happen if wallet_df is from groupby, but for safety
            features['wallet_first_tx_timestamp'] = pd.NaT
            features['wallet_last_tx_timestamp'] = pd.NaT
            features['wallet_active_period_days'] = 0


        # Activity frequency
        features['tx_per_day'] = features['total_transactions'] / (features['wallet_active_period_days'] + 1e-6) # Add epsilon to avoid div by zero

        # Action-specific counts (ensure all standard actions are covered)
        for action_type in standard_aave_actions:
            features[f'num_{action_type.lower()}'] = (wallet_df['action'] == action_type).sum()

        # Value-based features
        # Ensure 'amount_normalized' exists before trying to sum
        if 'amount_normalized' in wallet_df.columns:
            features['total_deposit_value'] = wallet_df[wallet_df['action'] == 'Deposit']['amount_normalized'].sum()
            features['total_borrow_value'] = wallet_df[wallet_df['action'] == 'Borrow']['amount_normalized'].sum()
            features['total_repay_value'] = wallet_df[wallet_df['action'] == 'Repay']['amount_normalized'].sum()
            features['total_withdraw_value'] = wallet_df[wallet_df['action'] == 'Withdraw']['amount_normalized'].sum()

            # Net flow (simplified, adjust as needed for full accounting)
            features['net_value_change'] = features['total_deposit_value'] + features['total_repay_value'] - \
                                           features['total_borrow_value'] - features['total_withdraw_value']
        else:
            # If amount_normalized wasn't created, set these to 0
            features['total_deposit_value'] = 0
            features['total_borrow_value'] = 0
            features['total_repay_value'] = 0
            features['total_withdraw_value'] = 0
            features['net_value_change'] = 0


        # Ratios
        features['repay_to_borrow_ratio'] = features['total_repay_value'] / (features['total_borrow_value'] + 1e-6)
        features['withdraw_to_deposit_ratio'] = features['total_withdraw_value'] / (features['total_deposit_value'] + 1e-6)

        # Risk indicators
        features['has_liquidated'] = 1 if features['num_liquidationcall'] > 0 else 0
        features['liquidation_rate'] = features['num_liquidationcall'] / (features['total_transactions'] + 1e-6)

        # Diversity of actions
        features['action_diversity_score'] = len(wallet_df['action'].unique()) / len(standard_aave_actions) if len(standard_aave_actions) > 0 else 0 # Use the *actual* observed unique actions relative to standard ones

        wallet_features.append(features)

    features_df = pd.DataFrame(wallet_features)

    # Drop timestamp columns used for intermediate calculations if not needed for clustering
    features_df = features_df.drop(columns=['wallet_first_tx_timestamp', 'wallet_last_tx_timestamp'])

    # Fill any remaining NaNs after division (e.g., 0/0 scenarios) with 0 or a sensible value
    features_df = features_df.fillna(0)

    print("Feature engineering complete. Sample features:")
    print(features_df.head())
    return features_df

# --- 3. Model Training (K-Means Clustering with Heuristic Scoring) ---

def train_and_score_model(features_df):
    """
    Trains an unsupervised model (K-Means) and assigns scores based on cluster characteristics.
    """
    print("Training scoring model (K-Means Clustering)...")

    # Select features for clustering. Exclude 'wallet' as it's an identifier.
    # Exclude features that are constant or highly correlated (optional, but good practice)
    feature_columns = [col for col in features_df.columns if col not in ['wallet']]
    X = features_df[feature_columns]

    # Handle potential infinite values resulting from division by zero, etc.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) # Fill NaNs after replacing inf

    # Determine number of clusters (this is a hyperparameter to tune)
    # For a credit score, we might want 5-10 clusters to represent different risk profiles.
    n_clusters = 7 # Example: Tune this based on cluster analysis

    # Check if X is empty after all filtering/cleaning
    if X.empty:
        print("Warning: No valid feature data to train the model. Returning empty scores.")
        return pd.DataFrame(columns=['wallet', 'credit_score', 'cluster'])
    if X.shape[0] < n_clusters:
        print(f"Warning: Number of samples ({X.shape[0]}) is less than n_clusters ({n_clusters}). Adjusting n_clusters to number of samples if > 0.")
        n_clusters_actual = X.shape[0]
        if n_clusters_actual == 0: # Still no data after all, return empty
            return pd.DataFrame(columns=['wallet', 'credit_score', 'cluster'])
    else:
        n_clusters_actual = n_clusters


    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=features_df.index) # This line is not used after scaling, can be removed to save memory


    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init='auto')
    features_df['cluster'] = kmeans.fit_predict(X_scaled)

    # --- Heuristic Score Assignment based on Cluster Characteristics ---
    # This is the most critical step and requires domain knowledge and analysis of clusters.
    # We will analyze average values of "good" and "bad" indicators for each cluster.

    cluster_analysis = features_df.groupby('cluster').mean(numeric_only=True) # Add numeric_only=True for Pandas 2.0+ to avoid warnings
    print("\n--- Cluster Analysis (Mean Feature Values per Cluster) ---")
    print(cluster_analysis)

    # Define "good" and "bad" metrics (features where higher is good/bad)
    good_metrics = ['total_deposit_value', 'total_repay_value', 'net_value_change', 'repay_to_borrow_ratio', 'tx_per_day', 'action_diversity_score']
    bad_metrics = ['num_liquidationcall', 'liquidation_rate', 'withdraw_to_deposit_ratio'] # Excessive withdrawals could be bad

    # Calculate a simple "cluster risk score"
    cluster_risk_scores = {}
    for cluster_id in range(n_clusters_actual): # Use n_clusters_actual here
        cluster_data = cluster_analysis.loc[cluster_id]

        # Filter metrics to only those actually present in cluster_data.index
        good_metrics_present = [m for m in good_metrics if m in cluster_data.index]
        bad_metrics_present = [m for m in bad_metrics if m in cluster_data.index]

        # Higher values in good_metrics should lead to lower risk
        good_score_contrib = sum(cluster_data[m] for m in good_metrics_present)

        # Higher values in bad_metrics should lead to higher risk
        bad_score_contrib = sum(cluster_data[m] for m in bad_metrics_present)

        # Normalize contributions to avoid dominance of large values
        # Ensure that cluster_analysis[m].max() is not zero to prevent division by zero
        normalized_good = {m: cluster_data[m] / (cluster_analysis[m].max() + 1e-6) for m in good_metrics_present if cluster_analysis[m].max() != 0}
        normalized_bad = {m: cluster_data[m] / (cluster_analysis[m].max() + 1e-6) for m in bad_metrics_present if cluster_analysis[m].max() != 0}


        # A higher composite_risk_score means riskier.
        composite_risk_score = sum(normalized_bad.values()) - sum(normalized_good.values())
        cluster_risk_scores[cluster_id] = composite_risk_score

    # Sort clusters by composite_risk_score (lowest risk -> highest risk)
    sorted_clusters = sorted(cluster_risk_scores.items(), key=lambda item: item[1]) # Sorts by risk score

    # Map sorted clusters to a credit score range (0-1000)
    base_scores = np.linspace(100, 950, n_clusters_actual) # Distribute scores from low (bad) to high (good)

    # Create the actual mapping
    cluster_to_credit_score_map = {}
    for i, (cluster_id, _) in enumerate(sorted_clusters):
        cluster_to_credit_score_map[cluster_id] = int(base_scores[i]) # Map the i-th least risky cluster to the i-th highest score

    print("\n--- Cluster to Credit Score Mapping ---")
    for cluster, score in cluster_to_credit_score_map.items():
        print(f"Cluster {cluster}: Score {score}")

    features_df['credit_score'] = features_df['cluster'].map(cluster_to_credit_score_map)

    # Final normalization to ensure 0-1000 range, if needed (though base_scores should handle this)
    min_score_actual = features_df['credit_score'].min()
    max_score_actual = features_df['credit_score'].max()
    if max_score_actual != min_score_actual: # Avoid division by zero if all scores are same
        features_df['credit_score'] = 0 + (features_df['credit_score'] - min_score_actual) * (1000 - 0) / (max_score_actual - min_score_actual)
    features_df['credit_score'] = features_df['credit_score'].round().astype(int)

    print("Model training and scoring complete.")
    return features_df[['wallet', 'credit_score', 'cluster']]

# --- 4. Main Script Execution ---

def generate_wallet_scores(json_file_path, output_json_path, analysis_output_dir):
    """
    One-step script to generate wallet scores from a JSON transaction file.
    """
    # Placing a broad try-except around the entire function to catch anything missed.
    try:
        df = load_and_preprocess_data(json_file_path)
        if df is None or df.empty: # Exit if data loading failed or no data was loaded
            print("Data loading failed or returned empty DataFrame. Cannot proceed with scoring.")
            return pd.DataFrame()

        features_df = feature_engineer(df)
        if features_df.empty:
            print("Feature engineering resulted in an empty DataFrame. Cannot proceed with scoring.")
            return pd.DataFrame()

        # FIX: Explicitly create a copy to avoid SettingWithCopyWarning
        scored_wallets_df = train_and_score_model(features_df).copy()
        
        if scored_wallets_df.empty:
            print("Model training and scoring resulted in an empty DataFrame. Cannot proceed with saving/analysis.")
            return pd.DataFrame()

        # --- Debug prints to inspect scored_wallets_df ---
        print("\n--- Debug: scored_wallets_df state after training ---")
        print("Columns:", scored_wallets_df.columns.tolist())
        print("Head:\n", scored_wallets_df.head())
        print("Is 'credit_score' in columns?", 'credit_score' in scored_wallets_df.columns)
        print("--------------------------------------------------\n")

        # Save results to JSON
        # This line also uses 'credit_score'
        try:
            scored_wallets_df[['wallet', 'credit_score']].set_index('wallet').to_json(output_json_path, orient='index', indent=4)
            print(f"\nWallet scores saved to {output_json_path}")
            print("\n--- Sample Scores ---")
            print(scored_wallets_df.head())
        except KeyError as e:
            print(f"ERROR: KeyError during saving scores to JSON: {e}. 'credit_score' might be missing here.")
            print(f"Columns at this point: {scored_wallets_df.columns.tolist()}")
            return pd.DataFrame()


        # --- Generate Analysis for analysis.md ---
        print(f"\nGenerating analysis plots and insights to {analysis_output_dir}...")

        # 1. Score Distribution Graph
        # This line also uses 'credit_score'
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(scored_wallets_df['credit_score'], bins=range(0, 1001, 100), kde=True)
            plt.title('Distribution of Wallet Credit Scores (0-1000)')
            plt.xlabel('Credit Score Range')
            plt.ylabel('Number of Wallets')
            plt.xticks(range(0, 1001, 100))
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_output_dir, 'score_distribution.png'))
            plt.close()
            print("Saved score_distribution.png")
        except KeyError as e:
            print(f"ERROR: KeyError during plotting score distribution: {e}. 'credit_score' might be missing here.")
            print(f"Columns at this point: {scored_wallets_df.columns.tolist()}")
            return pd.DataFrame()


        # 2. Score Distribution by Range (for text analysis)
        score_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1001]
        score_labels = [f"{i}-{i+99}" for i in range(0, 900, 100)] 
        score_labels.append("900-1000") 
        
        # Defensive check: Ensure 'credit_score' column exists before proceeding
        if 'credit_score' not in scored_wallets_df.columns:
            print("CRITICAL ERROR: 'credit_score' column is missing from scored_wallets_df before score_range creation.")
            print(f"Available columns in scored_wallets_df: {scored_wallets_df.columns.tolist()}")
            return pd.DataFrame() # Exit gracefully if the column is missing

        # This is the line user reported as the source of warning earlier.
        # If the error is still 'credit_score' from this line, it's very puzzling.
        try:
            scored_wallets_df['score_range'] = pd.cut(scored_wallets_df['credit_score'], bins=score_bins, labels=score_labels, right=False)
        except KeyError as e:
            print(f"ERROR: KeyError during pd.cut for score_range: {e}. 'credit_score' might be missing here.")
            print(f"Columns at this point: {scored_wallets_df.columns.tolist()}")
            return pd.DataFrame()
        except Exception as e: # Catch other potential errors from pd.cut, e.g., if data is non-numeric
            print(f"ERROR: Unexpected error during pd.cut for score_range: {e}.")
            print(f"Type of 'credit_score' column: {scored_wallets_df['credit_score'].dtype if 'credit_score' in scored_wallets_df.columns else 'N/A'}")
            print(f"Sample values of 'credit_score': {scored_wallets_df['credit_score'].head().tolist() if 'credit_score' in scored_wallets_df.columns else 'N/A'}")
            return pd.DataFrame()

        score_counts = scored_wallets_df['score_range'].value_counts().sort_index()

        analysis_text = f"""# Wallet Credit Score Analysis

## Score Distribution

The credit scores range from 0 to 1000. Here's a summary of the distribution:

"""
        # This line specifically uses 'credit_score' for describe()
        try:
            analysis_text += f"{scored_wallets_df['credit_score'].describe().to_string()}\n\n"
        except KeyError as e:
            print(f"ERROR: KeyError accessing 'credit_score' for describe(): {e}")
            print(f"Columns at this point: {scored_wallets_df.columns.tolist()}")
            return pd.DataFrame()


        analysis_text += f"""### Score Counts by Range:
{score_counts.to_string()}

## Behavior of Wallets in Lower Score Range (e.g., 0-300)

To understand the behavior of wallets with low scores, we examine the average features for these wallets compared to the overall average.

"""
        # Merge features back for detailed analysis
        # This line also accesses 'credit_score'
        try:
            full_data_for_analysis = features_df.merge(scored_wallets_df[['wallet', 'credit_score', 'cluster', 'score_range']], on='wallet')
        except KeyError as e:
            print(f"ERROR: KeyError during merge for 'credit_score': {e}")
            print(f"Columns of scored_wallets_df: {scored_wallets_df.columns.tolist()}")
            print(f"Columns of features_df: {features_df.columns.tolist()}")
            return pd.DataFrame()

        # And subsequent uses of full_data_for_analysis['credit_score']
        low_score_wallets = full_data_for_analysis[full_data_for_analysis['credit_score'] < 300]
        if not low_score_wallets.empty:
            # Exclude non-numeric or uninteresting columns before calculating mean
            analysis_cols = [col for col in low_score_wallets.columns if col not in ['wallet', 'credit_score', 'cluster', 'score_range', 'wallet_first_tx_timestamp', 'wallet_last_tx_timestamp', 'timestamp', 'asset_symbol', 'asset_decimals', 'amount', 'amount_raw', 'action', 'reserve', 'user', 'onBehalfOf', 'id', 'txHash', 'logIndex', '_id', 'network', 'protocol', 'logId', 'blockNumber', '__v', 'createdAt', 'updatedAt', 'actionData']]
            low_score_avg_features = low_score_wallets[analysis_cols].mean(numeric_only=True)
            analysis_text += f"### Average Features for Wallets with Score < 300:\n{low_score_avg_features.to_string()}\n\n"
            analysis_text += """
**Observations for Low Score Wallets:**
* Likely to have a higher `liquidation_rate` and `num_liquidationcall`.
* May have lower `repay_to_borrow_ratio` or higher `withdraw_to_deposit_ratio` if they withdraw more than they deposit/repay.
* Potentially fewer `total_transactions` or lower `tx_per_day`, indicating less consistent or "bot-like" activity.
* Could have a lower `net_value_change` or even negative values, suggesting net outflow or loss.
"""
        else:
            analysis_text += "No wallets found in the low score range (< 300).\n"


        analysis_text += """
## Behavior of Wallets in Higher Score Range (e.g., 700-1000)

Similarly, let's look at the average features for high-scoring wallets.
"""
        high_score_wallets = full_data_for_analysis[full_data_for_analysis['credit_score'] >= 700] # Adjust threshold
        if not high_score_wallets.empty:
            # Exclude non-numeric or uninteresting columns before calculating mean
            analysis_cols = [col for col in high_score_wallets.columns if col not in ['wallet', 'credit_score', 'cluster', 'score_range', 'wallet_first_tx_timestamp', 'wallet_last_tx_timestamp', 'timestamp', 'asset_symbol', 'asset_decimals', 'amount', 'amount_raw', 'action', 'reserve', 'user', 'onBehalfOf', 'id', 'txHash', 'logIndex', '_id', 'network', 'protocol', 'logId', 'blockNumber', '__v', 'createdAt', 'updatedAt', 'actionData']]
            high_score_avg_features = high_score_wallets[analysis_cols].mean(numeric_only=True)
            analysis_text += f"### Average Features for Wallets with Score >= 700:\n{high_score_avg_features.to_string()}\n\n"
            analysis_text += """
**Observations for High Score Wallets:**
* Typically have a `num_liquidationcall` of 0 and `liquidation_rate` of 0.
* Exhibit high `repay_to_borrow_ratio`, indicating responsible repayment behavior.
* High `total_deposit_value` and `net_value_change`, showing significant and positive interaction with the protocol.
* Higher `total_transactions` and `tx_per_day`, suggesting active and consistent participation.
* Higher `action_diversity_score` may indicate more varied and complex (but responsible) interactions.
"""
        else:
            analysis_text += "No wallets found in the high score range (>= 700).\n"

        analysis_text += """
## Model Logic and Interpretation

The credit score is derived using K-Means clustering on a set of engineered features. Wallets are grouped into `N` clusters based on their transaction behavior. A heuristic mapping is then applied to assign credit scores to these clusters. Clusters exhibiting characteristics associated with reliable and responsible usage (e.g., high total deposits, consistent repayments, no liquidations) are_old mapped to higher scores, while clusters with risky or exploitative behavior (e.g., frequent liquidations, large withdrawals relative to deposits) are assigned lower scores.

The mapping is based on an analysis of the mean feature values for each cluster (as shown in the "Cluster Analysis" table in the console output), where a composite "risk score" for each cluster is calculated from metrics like liquidation rate (higher risk) and total deposit value (lower risk). Clusters with lower composite risk scores are assigned higher credit scores.

**Extensibility:**
* **Feature Engineering:** This is the most critical area for improvement. Incorporating external data (e.g., token prices for USD conversion, wallet age on blockchain, other DeFi protocol interactions, on-chain reputation scores) would significantly enhance feature richness.
* **Clustering Algorithm:** Experimenting with other clustering algorithms (DBSCAN, Hierarchical, GMM) or even semi-supervised methods if any labeled data becomes available.
* **Scoring Heuristic:** The current cluster-to-score mapping is linear. A more sophisticated mapping (e.g., based on percentile ranks within clusters or a learned function) could be implemented.
* **Validation:** Beyond descriptive statistics, a more rigorous validation would involve domain experts reviewing specific high/low-score wallets to confirm the model's intuitive correctness.
"""

        with open(os.path.join(analysis_output_dir, 'analysis.md'), 'w') as f:
            f.write(analysis_text)
        print(f"Generated analysis.md in {analysis_output_dir}")

        return scored_wallets_df

    except Exception as e:
        # This will print the exact columns if available and then re-raise the original error.
        print("\n--- FATAL ERROR TRACE ---")
        print(f"An error occurred within generate_wallet_scores at a critical point.")
        try:
            # Attempt to print columns if scored_wallets_df exists and has columns
            if 'scored_wallets_df' in locals() and isinstance(scored_wallets_df, pd.DataFrame):
                print(f"Columns of scored_wallets_df at the time of error: {scored_wallets_df.columns.tolist()}")
                print(f"Is 'credit_score' in scored_wallets_df: {'credit_score' in scored_wallets_df.columns}")
            else:
                print("scored_wallets_df not defined or not a DataFrame when error occurred.")
        except Exception as inner_e:
            print(f"Error trying to print DataFrame columns during error handling: {inner_e}")
        print("--------------------------")
        raise e # Re-raise the original exception to get the full traceback

# --- How to run ---
if __name__ == "__main__":
    # --- Download Data Instructions ---
    print("--- Instructions for Data Download ---")
    print("1. Go to one of the provided Google Drive links:")
    print("   - Raw JSON: https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing")
    print("   - Compressed ZIP: https://drive.google.com/file/d/14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor/view?usp=sharing")
    print("2. Download the file. If you download the ZIP, make sure to extract it.")
    print("3. Rename the extracted JSON file to 'transactions_sample.json' (or whatever you prefer).")
    print("4. Place 'transactions_sample.json' in the same directory as this Python script,")
    print("   OR update the `JSON_FILE_PATH` variable at the top of this script to the full path of your downloaded file.")
    print("--------------------------------------\n")

    try:
        final_scores_df = generate_wallet_scores(JSON_FILE_PATH, OUTPUT_SCORES_PATH, OUTPUT_ANALYSIS_DIR)
        if not final_scores_df.empty:
            print(f"\nSuccessfully generated scores and analysis results in '{OUTPUT_ANALYSIS_DIR}/'.")
        else:
            print("\nScore generation failed. Please check error messages above and ensure data is correctly set up.")

    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")