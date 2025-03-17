import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import argparse
import platform

def get_params_from_head(cols, head):
    params = []
    for col in cols:
        if col.startswith(head):  # e.g., ATTENTIVE, GRU, LINEAR
            params.append(col)
    return params

def get_hyper_cols(cols, head):
    params = get_params_from_head(cols, head)
    fix_params = [
        'optimizer',
        'learning_rate',
        'criterion',
        'init_network',
        'reg_lambda',
        'reg_loss',
        'batch_size_training',
    ]
    return fix_params + params

# Load the grid search results from the CSV file
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='summary.csv', help='Path to the CSV file containing the grid search results.')
parser.add_argument('--target_col', type=str, default='mean_test_accuracy', help='Name of the column that holds the target performance metric.')
args = parser.parse_args()
csv_path = args.csv_path
target_col = args.target_col

df = pd.read_csv(csv_path)

# Process each head type separately
head_unique = df['head'].unique().tolist()
for head in head_unique:
    df_by_head = df[df['head'] == head]
    cols = df_by_head.columns

    # Choose a subset of columns that represent hyperparameters.
    hyperparameter_cols = get_hyper_cols(cols, head)
    selected_hyperparams = [col for col in hyperparameter_cols if col in df_by_head.columns]
    
    # Create a copy of the DataFrame for hyperparameters
    X = df_by_head[selected_hyperparams].copy()
    
    # Attempt to convert each hyperparameter to numeric if possible.
    # Columns that cannot be converted are considered categorical.
    numeric_cols = []
    non_numeric_cols = []
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
            numeric_cols.append(col)
        except Exception:
            non_numeric_cols.append(col)
    
    # One-hot encode only the non-numeric (categorical) columns.
    if non_numeric_cols:
        X_non_numeric = pd.get_dummies(X[non_numeric_cols], drop_first=False)
        # Combine the numeric columns (kept as-is) with the encoded categorical ones.
        X = pd.concat([X[numeric_cols], X_non_numeric], axis=1)
    
    # Set the target variable
    y = df_by_head[target_col]
    
    # Split the data for training and evaluation of the surrogate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Train a Random Forest Regressor as the surrogate model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate surrogate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE:", mse)
    
    # Extract feature importances from the trained model
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Display the importance scores
    print(importance_df)
    
    # Plot the feature importances for a visual interpretation
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f"Hyperparameter Importance from csv - {head} - {target_col}")
    plt.xlabel("Importance")
    plt.ylabel("Hyperparameter")
    plt.tight_layout()
    fig_path = os.path.split(csv_path)[0]
    plt.savefig(os.path.join(fig_path, f'_{head}hyper_importance_.png'))
    plt.close()
