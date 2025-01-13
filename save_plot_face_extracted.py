import matplotlib.pyplot as plt
import custom.tools as tools
import custom.faceExtractor as extractor
import numpy as np
import os
import pickle

face_extractor = extractor.FaceExtractor()
csv_array,cols = tools.get_array_from_csv(csv_path=os.path.join('partA','starting_point','samples.csv'))
list_subject_name = np.unique(csv_array[:,1])
face_ln_saving_folder = 'partA/video/mean_face_landmarks_per_subject'
saving_plot = os.path.join(face_ln_saving_folder,'plot_landmarks')
list_landmarks = []
count = 0

if not os.path.exists(saving_plot):
  os.makedirs(saving_plot)
if not os.path.exists(face_ln_saving_folder):
  os.makedirs(face_ln_saving_folder)
for subject in list_subject_name:
  count += 1
  with open(os.path.join(face_ln_saving_folder,subject+'.pkl'), 'rb') as f:
    dict_subject = pickle.load(f)
    mean_facial_landmarks = dict_subject['mean_facial_landmarks']
    mean_facial_landmarks = face_extractor.convert_from_numpy_to_NormalizedLandmark(mean_facial_landmarks)
    black_img = np.zeros((256,256,3), dtype=np.uint8)
    annotated_image,_ = face_extractor.plot_landmarks(image=black_img, 
                                                landmarks=mean_facial_landmarks,
                                                connections=face_extractor.FACE_TESSELATION)
    plt.imshow(annotated_image[:,:,::-1]/255)
    plt.title(subject)
    plt.savefig(os.path.join(saving_plot,subject+'.png'))
    # plt.show()
  print(f'Saved {count}/{len(list_subject_name)} plots')  

# Plot mean landmarks of all subjects
subject = 'all_subjects_mean_landmarks'
with open(os.path.join(face_ln_saving_folder,subject+'.pkl'), 'rb') as f:
  dict_subject = pickle.load(f)
  mean_facial_landmarks = dict_subject['mean_facial_landmarks']
  mean_facial_landmarks = face_extractor.convert_from_numpy_to_NormalizedLandmark(mean_facial_landmarks)
  black_img = np.zeros((256,256,3), dtype=np.uint8)
  annotated_image,_ = face_extractor.plot_landmarks(image=black_img, 
                                                landmarks=mean_facial_landmarks,
                                                connections=face_extractor.FACE_TESSELATION)
  
  plt.imshow(annotated_image[:,:,::-1]/255)
  plt.title(subject)
  plt.savefig(os.path.join(saving_plot,subject+'.png'))
  print(f'{subject} saved in path {os.path.join(saving_plot,subject+".png")}')
  # plt.show()