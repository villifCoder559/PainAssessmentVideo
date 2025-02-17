import custom.tools as tools
import custom.scripts as scripts
import os
import time
# OK finish the video clip, start to create plot list_same_clip_positions many people
# try to combine plot and video in one

folder_tsne_results = os.path.join('tsne_Results_frontalized',f'test_{str(int(time.time()))}')
folder_path_features = os.path.join('partA','video','features','samples_16_frontalized')

if not os.path.exists(folder_tsne_results):
    os.makedirs(folder_tsne_results)

stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
all_subjects = list(range(1,88))
# subject_id_list = [sbj for sbj in all_subjects if sbj not in stoic_subjects]
subject_id_list = stoic_subjects
count = 0
for sub in subject_id_list:  
  clip_list = [0,1,2,3,4,5,6,7]
  class_list = [4]
  sample_id_list = [84]
  sliding_windows =  16
  legend_label = 'clip' # can be clip, subject and class
  tools.plot_and_generate_video(folder_path_features=folder_path_features,
                                  folder_path_tsne_results=folder_tsne_results,
                                  subject_id_list=[sub],
                                  clip_list=clip_list,
                                  legend_label=legend_label,
                                  class_list=class_list,
                                  sliding_windows=sliding_windows,
                                  plot_only_sample_id_list=sample_id_list,
                                  plot_third_dim_time=False,
                                  create_video=False,
                                  apply_pca_before_tsne=True,
                                  tsne_n_component=2,
                                  sort_elements=True,
                                  cmap='copper') # copper, tab20
  count+=1
  print(f'Processed {count}/{len(subject_id_list)}')
  break
# save configuration in folder_tsne_results
with open(os.path.join(folder_tsne_results,'config.txt'),'w') as f:
    f.write(f'subject_id_list: {subject_id_list}\n')
    f.write(f'clip_list: {clip_list}\n')
    f.write(f'class_list: {class_list}\n')
    f.write(f'sliding_windows: {sliding_windows}\n')
    f.write(f'legend_label: {legend_label}\n')
    f.write(f'plot_third_dim_time: False\n')
    f.write(f'apply_pca_before_tsne: True\n')
    f.write(f'tsne_n_component: 2\n')
    f.write(f'sort_elements: True\n')
    f.write(f'cmap: copper\n')
    f.write(f'folder_path_features: {folder_path_features}\n')
    f.write(f'folder_path_tsne_results: {folder_tsne_results}\n')
    f.write(f'create_video: False\n')
print(f'Configuration saved in {os.path.join(folder_tsne_results,"config.txt")}')