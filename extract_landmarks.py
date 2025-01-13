import numpy as np
import custom.faceExtractor as extractor
import os
import time
import pickle
import custom.tools as tools

import numpy as np
import custom.faceExtractor as extractor
import os
import time
import cv2
import pickle
import custom.tools as tools
import matplotlib.pyplot as plt
# subject_name = '112016_m_25'

root_video_path = 'partA/video/video'
csv_array,cols = tools.get_array_from_csv(csv_path=os.path.join('partA','starting_point','samples.csv'))
list_subject_name = np.unique(csv_array[:,1])
face_ln_saving_folder = 'partA/video/mean_face_landmarks_per_subject'
if not os.path.exists(face_ln_saving_folder):
  os.makedirs(face_ln_saving_folder)
face_extractor = extractor.FaceExtractor()
start = time.time()
count = 0
all_mean_facial_landmarks = np.zeros((478,3),dtype=np.float32)

for subject_name in list_subject_name:
  folder_path = os.path.join(root_video_path, subject_name)
  all_videos = os.listdir(folder_path)
  list_video_path = [os.path.join(folder_path, video) for video in all_videos]
  mean_facial_landmarks,count_frame=face_extractor.get_mean_facial_landmarks(list_video_path=list_video_path,
                                                                                 align=True,
                                                                                 numpy_view=True)
  dict_subject = {'subject_name':subject_name,
                  'mean_facial_landmarks':mean_facial_landmarks,
                  'count_frame':count_frame}
  all_mean_facial_landmarks += mean_facial_landmarks
  with open(os.path.join(face_ln_saving_folder,subject_name+'.pkl'), 'wb') as f:
    pickle.dump(dict_subject, f)
  total_time = time.time()-start
  count += 1
  expected_time = total_time/count*len(list_subject_name)
  print('-'*30)
  print(f'{subject_name} saved in path {os.path.join(face_ln_saving_folder,subject_name+".pkl")}')
  print(f'Processed {count}/{len(list_subject_name)} subjects')
  print(f'Expected end: {int(expected_time/60/60)} h {int(expected_time/60)%60} m {int(expected_time)%60} s')
  print(f'Total time  : {int(total_time/60/60)} h {int(total_time/60)%60} m {int(total_time)%60} s')
  print('-'*30)
  
# Save mean landmarks of all subjects
all_mean_facial_landmarks /= len(list_subject_name)
dict_subject = {'mean_facial_landmarks':all_mean_facial_landmarks,
                'count_subject':len(list_subject_name)}
with open(os.path.join(face_ln_saving_folder,'all_subjects_mean_landmarks.pkl'), 'wb') as f:
  pickle.dump(dict_subject, f)
print(f'All subjects mean landmarks saved in {os.path.join(face_ln_saving_folder,"all_subjects_mean_landmarks.pkl")}')  

