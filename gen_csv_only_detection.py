import os 
import numpy as np
import custom.tools as tools 

path = os.path.join("partA","video","mean_face_landmarks_per_subject","no_detection_log.txt")
with open(path) as f:
  no_detection_log = f.readlines()
list_names = []
for line in no_detection_log:
  list_names.append(line.split(",")[0])
list_names = np.array(list_names)
unique = np.unique(list_names)
print(unique.shape)
unique
csv_path = os.path.join('partA','starting_point','samples.csv')
csv_array,cols = tools.get_array_from_csv(csv_path=csv_path)
new_csv = []
video_root_folder = os.path.join("partA","video","video") 
for el in csv_array:
  sample_path = os.path.join(video_root_folder,el[1],el[5]+'.mp4') 
  print(sample_path)
  print(unique[0])
  if sample_path not in unique:
    new_csv.append(el)
print(new_csv)
tools.generate_csv(cols=cols,data=new_csv,saving_path=os.path.join('partA','starting_point','samples_exc_no_detection.csv'))