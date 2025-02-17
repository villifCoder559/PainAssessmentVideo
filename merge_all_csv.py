
import os
import pandas as pd
import argparse
from custom.helper import GLOBAL_PATH,CSV
import time
def merge_summary_csvs(parent_folder, output_folder, output_file):
  all_dfs = []
  os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists
  output_path = os.path.join(output_folder, output_file)
  
  # Walk through all directories
  list_folder = os.listdir(parent_folder)
  for folder in list_folder:
    test_folder = os.path.join(parent_folder, folder)
    for _,_,files in os.walk(test_folder):
      if "summary_log.csv" in files: 
        file_path = os.path.join(test_folder, "summary_log.csv") # in each history_run there is a csv called summary_log
                                                                 # that summarize the results of all test in that history_run
        try:
          df = pd.read_csv(file_path)
          df["history_source"] = folder  # Add history source folder column
          all_dfs.append(df)
        except Exception as e:
          print(f"Error reading {file_path}: {e}")
  
  # Merge all dataframes
  if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    merged_df=merged_df[CSV.sort_cols + [col for col in df.columns if col not in CSV.sort_cols]]
    # sort by tot_test_accuracy
    merged_df = merged_df.sort_values(by='tot_test_accuracy',ascending=False)
    merged_df.to_csv(output_path+'.csv', index=False)
    print(f"Merged CSV saved to {output_path}+'.csv'")
  else:
    print("No summary.csv files found.")

if __name__ == "__main__":
  # current_time = time.strftime("%Y%m%d-%H%M%S")
  parser = argparse.ArgumentParser(description="Merge all summary.csv files from subdirectories into one CSV.")
  parser.add_argument("--parent_folder",required=True, help="Path to the parent folder containing subdirectories with summary.csv files")
  parser.add_argument("--output_folder",required=True, help="Path to the folder where the merged CSV should be saved")
  parser.add_argument("--output_file", default=f"merged_summary_log",help="Name of the output merged CSV file")
  parser.add_argument("--gp", action="store_true", help="Use global path")
  # scrpt example: python3 merge_all_csv.py --parent_folder PainAssessmentVideo/Tests  --output_folder PainAssessmentVideo/Tests/aSummary_csv
  args = parser.parse_args()
  if args.gp:
    args.parent_folder = os.path.join(GLOBAL_PATH.NAS_PATH, args.parent_folder)
    args.output_folder = os.path.join(GLOBAL_PATH.NAS_PATH, args.output_folder)
    print(f"Parent folder: {args.parent_folder}")
    print(f"Output folder: {args.output_folder}")
  merge_summary_csvs(args.parent_folder, args.output_folder, args.output_file)