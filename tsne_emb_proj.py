import openTSNE
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import custom.tools as tools
import tqdm
import custom.tsne_cuda_tools as tsne_tools

def load_feats(feat_path, df_source):
  dict_data = {}
  if tools.is_dict_data(feat_path):
    tmp = tools.load_dict_data(saving_folder_path=feat_path)
    for sample_id in df_source['sample_id'].unique():
      mask = sample_id == tmp['list_sample_id']
      for k, v in tmp.items():
        if k not in dict_data:
          dict_data[k] = []
        dict_data[k].append(v[mask])
  else:
    # Load features from the specified path
    for row in tqdm.tqdm(df_source.itertuples(index=False),desc="Loading features"):
      sbj_name = row.subject_name
      sample_name = row.sample_name
      feat_path = os.path.join(feat_path, sbj_name, f'{sample_name}.safetensors')
      tmp = tools.load_dict_data(feat_path)
      for k,v in tmp.items():
        if k not in dict_data:
          dict_data[k] = []
        dict_data[k].append(v) 
    
  # Convert lists to numpy arrays
  for k,v in dict_data.items():
    dict_data[k] = np.concatenate(v, axis=0)
  return dict_data
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Project embeddings using t-SNE")
  parser.add_argument("--csv_path", type=str, required=True, help="Path to the embeddings CSV file")
  parser.add_argument("--output_path", type=str, required=True, help="Path to save the t-SNE plot")
  parser.add_argument("--sample_id_origin", required=True, type=int, default="sample_id", help="Column name for sample IDs to use to create space embedding in the CSV file")
  parser.add_argument("--sample_id_proj", required=True, type=int, default="sample_id_projected", help="Column name for projected sample IDs in the CSV file")
  parser.add_argument("--feat_path", type=str, required=True, help="Path to the features ")
  parser.add_argument("--threshold", type=int, default=5, help="Threshold for binary plotting. Default is 5.")
  parser.add_argument("--stride_dataset", type=int, default=16, help="Stride used to compute features per windows")
  parser.add_argument("--fps_dataset", type=int, default=25, help="Frames per second for the videos used to compute features")
  parser.add_argument("--video_path", type=str, default=None, help="Path to the video files")
  
  args = parser.parse_args()
  dict_args = vars(args)
  
  # Generate df for source embedding space 
  df_source = pd.read_csv(dict_args["csv_path"],sep="\t")
  df_source = df_source[dict_args["sample_id_origin"] == df_source['sample_id']]
  if df_source.empty:
    raise ValueError(f"No data found for sample_id {dict_args['sample_id_origin']} in the CSV file.")
  
  # Generate df for elements to project
  df_projected = pd.read_csv(dict_args["csv_path"],sep="\t")
  df_projected = df_projected[dict_args["sample_id_proj"] == df_projected['sample_id']]
  if df_projected.empty:
    raise ValueError(f"No data found for sample_id {dict_args['sample_id_proj']} in the projected CSV file.")
  
  # load source data 
  dict_data_source = load_feats(dict_args["feat_path"], df_source)
      
  # Compute t-SNE on the source embedding space    
  tsne = openTSNE.TSNE(n_components=2,
                       perplexity=30,
                       metric="euclidean",
                       n_jobs=4,
                       random_state=42)
  
  embeddings = tsne.fit(dict_data_source['features'].reshape(-1, dict_data_source['features'].shape[-1]))

  dict_data_projected = load_feats(dict_args["feat_path"], df_projected)
  
  original_shape = []
  unique_sample_ids = np.unique(dict_data_projected['list_sample_id'])
  for sample in unique_sample_ids:
    mask = dict_data_projected['list_sample_id'] == sample
    original_shape.append(dict_data_projected['features'][mask].shape[:-1])
  
  # Project the new embeddings into the t-SNE space
  new_embeddings = embeddings.transform(dict_data_projected['features'].reshape(-1, dict_data_source['features'].shape[-1]))
  
  # Reshape to original shape
  tmp_reshape_list = []
  for sample_id,sample_shape in zip(unique_sample_ids, original_shape):
    mask = dict_data_projected['list_sample_id'] == sample_id
    mask = np.repeat(mask,np.prod(*sample_shape))
    tmp_reshape_list.append(new_embeddings[mask].reshape(*sample_shape, -1))
  dict_data_projected['features'] = np.concatenate(tmp_reshape_list, axis=0)

  # Plot results
  list_additional_desc_legend = []
  list_additional_desc_legend.append(f' (0, {dict_args["stride_dataset"]/dict_args["fps_dataset"]*dict_args["threshold"]}) sec')
  list_additional_desc_legend.append(f' ({dict_args["stride_dataset"]/dict_args["fps_dataset"]*dict_args["threshold"]}, {(dict_data_projected["features"].shape[0])*dict_args["stride_dataset"]/dict_args["fps_dataset"]:.2f}) sec')
  name_source = df_source['sample_name'].iloc[0]
  name_projected = df_projected['sample_name'].iloc[0]
  title = f"{name_projected}_sourceTSNE_{name_source}_{dict_data_projected['features'].shape[:-1]}_threshold_{dict_args['threshold']}"
  list_rgb_plot = []
  # dict_data_projected['features'] = dict_data_projected['features'].reshape(-1, dict_data_projected['features'].shape[-1])
  binary_threshold = np.arange(new_embeddings.shape[0],dtype=np.int32)
  binary_threshold = np.where(binary_threshold < dict_args["threshold"], 1, 0)
  binary_threshold = binary_threshold.reshape(*original_shape[0])
  for slc in tqdm.tqdm(range(1, dict_data_projected['features'].shape[0]+1), desc="Plotting t-SNE"):
    img = tsne_tools.plot_tsne(dict_data_projected['features'][:slc].reshape(-1, dict_data_projected['features'].shape[-1]),
                        binary_threshold[:slc],
                        list_additiona_desc_legend=list_additional_desc_legend,
                        saving_path=None,
                        legend_label='Movement',
                        title=title,
                        cmap='bwr',
                        )
    list_rgb_plot.append(img)
    
  # Generate video from t-SNE embeddings
  video_path = os.path.join(dict_args['video_path'], df_projected['subject_name'].iloc[0], f"{df_projected['sample_name'].iloc[0]}.mp4")
  tsne_tools.generate_video_from_list_video_path(video_path=video_path,
                                                list_clip_ranges=dict_data_projected['list_frames'],
                                                list_rgb_image_plot=list_rgb_plot,
                                                output_fps=4,
                                                sample_id=dict_args['sample_id_proj'],
                                                y_gt=df_projected['class_id'].iloc[0],
                                                saving_path=os.path.join(dict_args['output_path'],'tsne_video')
                                                )
  
  
  