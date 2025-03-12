import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def nested_dict():
  return defaultdict(nested_dict)

def cmp_per_subject_preprocessing(nested_dict_res):
  """
  Preprocesses a nested dictionary by computing the absolute distances between 
  the clusters in each dataset type for each subject and feature reduction method.
  """
  result_dict = nested_dict()
  for group,dict_subjects in nested_dict_res.items():
    for subject,dict_feat_reduction in dict_subjects.items():
      for feat_reduction,dict_dataset_type in dict_feat_reduction.items():
        # compute the distances between the clusters in each dataset type
        list_values = list(dict_dataset_type.values())
        list_keys = list(dict_dataset_type.keys())
        for i in range(len(list_values)):
          for j in range(i+1,len(list_values)):
            result_dict[group][subject][feat_reduction][list_keys[i]+'|'+list_keys[j]] = abs(list_values[i]-list_values[j])
  return result_dict         
def cmp_per_subject_feat_reduction(nested_dict_res):
  result_dict = nested_dict()
  for group,dict_subjects in nested_dict_res.items():
    for subject,dict_feat_reduction in dict_subjects.items():
      list_feat_reduction_key = list(dict_feat_reduction.keys())
      for i in range(len(list_feat_reduction_key)):
        for j in range(i+1,len(list_feat_reduction_key)):
          list_key_i = list(dict_feat_reduction[list_feat_reduction_key[i]].keys())
          list_key_j = list(dict_feat_reduction[list_feat_reduction_key[j]].keys())
          intersection_keys = list(set(list_key_i).intersection(list_key_j))
          for key in intersection_keys:
            result_dict[group][subject][list_feat_reduction_key[i]+'|'+list_feat_reduction_key[j]][key] = abs(dict_feat_reduction[list_feat_reduction_key[i]][key]-dict_feat_reduction[list_feat_reduction_key[j]][key])
  return result_dict  

def plot_preprocessing_results(df,X,Y,HUE,output_file):
  """
  Plots the preprocessing results in a bar chart.
  """
  plt.figure(figsize=(20, 10))
  sns.set_theme(style="whitegrid")
  filtered_df = df.copy()
  scaler = MinMaxScaler()
  filtered_df["abs_diff_normalized"] = scaler.fit_transform(filtered_df[["abs_diff"]])
  filtered_df = filtered_df[[X,Y,HUE]]
  sns.scatterplot(
    data=filtered_df,
    x=X,
    y=Y,
    hue=HUE,
    palette="tab10",
    alpha=0.7,
    s=200
  )
  plt.grid(axis='y')
  plt.xlabel(X)
  plt.ylabel(Y)
  plt.title(f"{HUE} Performance Based on {Y}")
  plt.legend(title=HUE, loc="best")
  plt.savefig(output_file)
  plt.close()
  
def get_pd_dataframe(dict_results):
  """
  Converts a nested dictionary into a pandas DataFrame.
  """
  # Flatten the nested defaultdict into a list of records
  flattened_data = []
  for group, ids in dict_results.items(): # group_0,...
    for id_key, metrics in ids.items(): # id_1, id_2,...
      for metric_type, comparisons in metrics.items(): # mean,temporal
        for comparison, value in comparisons.items(): # samples_16|samples_16_aligned_cropped,...
          flattened_data.append({
            "group": group,
            "id": id_key.split('_')[1],
            "metric": metric_type,
            "comparison": comparison,
            "abs_diff": value
          })
  # Convert to DataFrame
  df = pd.DataFrame(flattened_data)
  # df.to_csv("comparison_results.csv", index=False)
  # print(df)
  return df
# Display the DataFrame


if __name__ == '__main__':
  nested_dict_path = os.path.join("tsne_Results_all_grouped/test_1740400199/nested_dict_results.pkl")
  with open(nested_dict_path, 'rb') as f:
    nested_dict_results = pickle.load(f)
  # print(nested_dict_results)
  dict_preprocessing_result = cmp_per_subject_preprocessing(nested_dict_results)
  dict_feat_reduction_result = cmp_per_subject_feat_reduction(nested_dict_results)
  df = get_pd_dataframe(dict_preprocessing_result)
  output_folder = 'comparison_results'
  df.to_csv(os.path.join(output_folder,'comparison_results.csv'), index=False)
  # get only df with group_0 and mean
  for group,_ in dict_preprocessing_result.items():
    df_filtered = df[  (df['group'] == group)  
                    & (df['metric'] == 'mean') 
            # & (df['comparison'] == 'samples_16_aligned_cropped|samples_16_frontalized')
          ]
    if not os.path.exists(output_folder):
      os.makedirs(output_folder, exist_ok=True)
    plot_preprocessing_results(df_filtered,'id','abs_diff_normalized','comparison',os.path.join(output_folder,f'{group}_mean.png'))
  # plot_preprocessing_results(dict_preprocessing_result)
