import custom.tools as tools
import custom.scripts as scripts
import os
import time
import matplotlib.pyplot as plt
# OK finish the video clip, start to create plot list_same_clip_positions many people
# try to combine plot and video in one
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def generate_unique_plot(list_tsne_plot_per_folder_feature):
  fig, axs = plt.subplots(1, len(list_tsne_plot_per_folder_feature), figsize=(15, 5))
  for i, ax in enumerate(list_tsne_plot_per_folder_feature):
    axs[i]=ax
  plt.tight_layout()
  plt.savefig(os.path.join(folder_tsne_results,f'plot_merged.png'))
  # return fig,axs


folder_tsne_results = os.path.join('tsne_Results_per_sbj',f'test_{str(int(time.time()))}')
# folder_path_features = os.path.join('partA','video','features','samples_16_frontalized')
list_folder_path_features = [os.path.join('partA','video','features','samples_16'),
                             os.path.join('partA','video','features','samples_16_aligned_cropped'),
                             os.path.join('partA','video','features','samples_16_frontalized')]

if not os.path.exists(folder_tsne_results):
    os.makedirs(folder_tsne_results)

# stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
all_subjects = list(range(1,88))
# subject_id_list = [sbj for sbj in all_subjects if sbj not in stoic_subjects]
# subject_id_list = stoic_subjects
all_subjects = [list(range(0,20)),list(range(20,40)),list(range(40,60)),list(range(60,80)),list(range(80,88))]
count = 0

# for sub in subject_id_list:
list_tsne_plot_per_folder_feature = []  
clip_list = [0,1,2,3,4,5,6,7]
class_list = [0,1,2,3,4]
# sample_id_list = [84]
sliding_windows =  16
legend_label = 'subject' # can be clip, subject and class
for chunk_sbj in all_subjects:
  fig, axes = plt.subplots(len(list_folder_path_features), 1, figsize=(20, 20))
  for i,folder_path_features in enumerate(list_folder_path_features):
    dict_plot = tools.plot_and_generate_video(folder_path_features=folder_path_features,
                                    folder_path_tsne_results=folder_tsne_results,
                                    subject_id_list=chunk_sbj,
                                    clip_list=clip_list,
                                    legend_label=legend_label,
                                    class_list=class_list,
                                    sliding_windows=sliding_windows,
                                    # plot_only_sample_id_list=sample_id_list,
                                    plot_third_dim_time=False,
                                    create_video=False,
                                    apply_pca_before_tsne=True,
                                    tsne_n_component=2,
                                    sort_elements=True,
                                    cmap='tab20',
                                    ) # copper, tab20
    dict_plot['saving_path'] = None
    tools.plot_tsne(**dict_plot,ax=axes[i],return_ax=True)
      # list_tsne_plot_per_folder_feature.append(tools.plot_tsne(**dict_plot,ax=axes[i],return_ax=True))
    # plt.savefig(os.path.join(folder_tsne_results,f'plot_merged.png'))
    
    #save plot
  plt.savefig(os.path.join(folder_tsne_results,f'plot_{chunk_sbj}.png'))
  plt.close(fig)
  count+=1
  # print(f'Processed {count}/{len(subject_id_list)}')
  print(f'Processed {count}/{len(all_subjects)}')
  
# save configuration in folder_tsne_results
with open(os.path.join(folder_tsne_results,'config.txt'),'w') as f:
    f.write(f'subject_id_list: {all_subjects}\n')
    f.write(f'clip_list: {clip_list}\n')
    f.write(f'class_list: {class_list}\n')
    f.write(f'sliding_windows: {sliding_windows}\n')
    f.write(f'legend_label: {legend_label}\n')
    f.write(f'plot_third_dim_time: False\n')
    f.write(f'apply_pca_before_tsne: True\n')
    f.write(f'tsne_n_component: 2\n')
    f.write(f'sort_elements: True\n')
    f.write(f'cmap: copper\n')
    f.write(f'folder_path_features: {list_folder_path_features}\n')
    f.write(f'folder_path_tsne_results: {folder_tsne_results}\n')
    f.write(f'create_video: False\n')
print(f'Configuration saved in {os.path.join(folder_tsne_results,"config.txt")}')