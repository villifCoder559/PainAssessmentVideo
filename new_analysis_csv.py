import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
def old_analysis_csv(file_path,output_folder_plot):
  # Load CSV file from command-line argument
  if len(sys.argv) < 2:
    print("Usage: python analyze_results.py <path_to_csv>")
    sys.exit(1)

  file_path = sys.argv[1]
  df = pd.read_csv(file_path)
  if not os.path.exists(output_folder_plot):
    os.makedirs(output_folder_plot)
  # Compute mean test accuracy and loss across k0, k1, k2
  df["mean_test_accuracy"] = df[["test_accuracy_k0", "test_accuracy_k1", "test_accuracy_k2"]].mean(axis=1)
  df["mean_test_loss"] = df[["test_loss_k0", "test_loss_k1", "test_loss_k2"]].mean(axis=1)

  # Get top 5 configurations by accuracy and lowest test loss
  top_by_accuracy = df.nlargest(5, "mean_test_accuracy")[["test_id", "model", "optimizer", "learning_rate", "mean_test_accuracy", "mean_test_loss"]]
  top_by_loss = df.nsmallest(5, "mean_test_loss")[["test_id", "model", "optimizer", "learning_rate", "mean_test_accuracy", "mean_test_loss"]]

  print("\nTop 5 Configurations by Test Accuracy:")
  print(top_by_accuracy)

  print("\nTop 5 Configurations by Lowest Test Loss:")
  print(top_by_loss)

  # Set seaborn style
  sns.set_theme(style="whitegrid")

  # Learning Rate vs Mean Test Accuracy
  plt.figure(figsize=(8, 5))
  sns.scatterplot(x=df["learning_rate"], y=df["mean_test_accuracy"], hue=df["optimizer"], alpha=0.7)
  plt.xscale("log")
  plt.xlabel("Learning Rate (log scale)")
  plt.ylabel("Mean Test Accuracy")
  plt.title("Impact of Learning Rate on Test Accuracy")
  plt.legend(title="Optimizer")
  plt.savefig(os.path.join(output_folder_plot,"learning_rate_vs_accuracy.png"))
  plt.close()

  # Batch Size vs Mean Test Accuracy
  plt.figure(figsize=(8, 5))
  sns.boxplot(x=df["batch_size_training"], y=df["mean_test_accuracy"])
  plt.xlabel("Batch Size (Training)")
  plt.ylabel("Mean Test Accuracy")
  plt.title("Effect of Batch Size on Test Accuracy")
  plt.savefig(os.path.join(output_folder_plot,"batch_size_vs_accuracy.png"))
  plt.close() 

  # GRU Hidden Size vs Mean Test Accuracy
  plt.figure(figsize=(8, 5))
  sns.scatterplot(x=df["GRU.hidden_size"], y=df["mean_test_accuracy"], alpha=0.7)
  plt.xlabel("GRU Hidden Size")
  plt.ylabel("Mean Test Accuracy")
  plt.title("Impact of GRU Hidden Size on Test Accuracy")
  plt.savefig(os.path.join(output_folder_plot,"gru_hidden_size_vs_accuracy.png"))
  plt.close()

  # Compute mean training, validation, and test loss
  df["mean_train_loss"] = df[["train_loss_k0", "train_loss_k1", "train_loss_k2"]].mean(axis=1)
  df["mean_val_loss"] = df[["val_loss_k0", "val_loss_k1", "val_loss_k2"]].mean(axis=1)

  # Training Loss vs Validation Loss
  plt.figure(figsize=(8, 5))
  sns.scatterplot(x=df["mean_train_loss"], y=df["mean_val_loss"], alpha=0.7)
  plt.xlabel("Mean Training Loss")
  plt.ylabel("Mean Validation Loss")
  plt.title("Training Loss vs Validation Loss (Overfitting Check)")
  plt.savefig(os.path.join(output_folder_plot,"train_loss_vs_val_loss.png"))
  plt.close()

  # Validation Loss vs Test Loss
  plt.figure(figsize=(8, 5))
  sns.scatterplot(x=df["mean_val_loss"], y=df["mean_test_loss"], alpha=0.7)
  plt.xlabel("Mean Validation Loss")
  plt.ylabel("Mean Test Loss")
  plt.title("Validation Loss vs Test Loss (Generalization Check)")
  plt.savefig(os.path.join(output_folder_plot,"val_loss_vs_test_loss.png"))
  plt.close()
  
def new_analysis_csv(file_path,output_folder_plot):
  # === USER SETTINGS ===
  FILTERS = {
    # "batch_size_training": 32,  # Example: Uncomment to filter by batch size
  }

  # Define columns to analyze
  X_COLUMN = "learning_rate"       # X-axis (numeric, e.g., learning_rate)
  Y_COLUMN = "mean_test_accuracy"      # Y-axis (numeric, e.g., test loss or accuracy)
  HUE_COLUMN = "GRU.layer_norm"         # Categorical variable (color-coded, e.g., optimizer)

  # === LOAD DATA ===
  if len(sys.argv) < 2:
    print("Usage: python analyze_results.py <path_to_csv>")
    sys.exit(1)

  file_path = sys.argv[1]
  df = pd.read_csv(file_path)

  # Compute additional metrics if needed
  if "mean_test_loss" not in df.columns:
    df["mean_test_loss"] = df[["test_loss_k0", "test_loss_k1", "test_loss_k2"]].mean(axis=1)

  if "mean_test_accuracy" not in df.columns:
    df["mean_test_accuracy"] = df[["test_accuracy_k0", "test_accuracy_k1", "test_accuracy_k2"]].mean(axis=1)
  
  if "mean_train_loss" not in df.columns:
    df["mean_train_loss"] = df[["train_loss_k0", "train_loss_k1", "train_loss_k2"]].mean(axis=1)
  
  if "mean_val_loss" not in df.columns:
    df["mean_val_loss"] = df[["val_loss_k0", "val_loss_k1", "val_loss_k2"]].mean(axis=1)
  # === FILTER DATA ===
  filtered_df = df.copy()
  for key, value in FILTERS.items():
    if key in df.columns:
      filtered_df = filtered_df[filtered_df[key] == value]

  if filtered_df.empty:
    print(f"No rows found matching filters: {FILTERS}")
    sys.exit(0)

  print(f"\nFiltered data (matching {FILTERS}):")
  print(filtered_df[[X_COLUMN, Y_COLUMN, HUE_COLUMN]])

  # === PLOT RESULTS ===
  plt.figure(figsize=(8, 5))
  sns.set_style("whitegrid")

  # Scatter plot with Seaborn
  sns.scatterplot(
    data=filtered_df,
    x=X_COLUMN,
    y=Y_COLUMN,
    hue=str(HUE_COLUMN),
    palette="tab10",
    alpha=0.7
  )

  # Adjust plot settings
  if X_COLUMN in ["learning_rate", "reg_lambda"]:
    plt.xscale("log")  # Log scale for better visualization

  plt.xlabel(X_COLUMN.replace("_", " ").title())
  plt.ylabel(Y_COLUMN.replace("_", " ").title())
  title = f"{HUE_COLUMN.replace('_', ' ').title()} Performance Based on {Y_COLUMN.replace('_', ' ').title()}"
  plt.title(title)
  plt.legend(title=HUE_COLUMN.replace("_", " ").title(), loc="best")
  png_file = os.path.join(output_folder_plot,f"{X_COLUMN},{Y_COLUMN},{HUE_COLUMN},{list(FILTERS.keys())}.png")
  plt.savefig(png_file)
  plt.close()
  print(f"Plot saved in {png_file}")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python analyze_filtered_results.py <path_to_csv>")
    sys.exit(1)
  file_path = sys.argv[1]
  output_folder_plot = os.path.join(os.path.split(file_path)[0],'csv_plots')
  new_analysis_csv(file_path,output_folder_plot)
  