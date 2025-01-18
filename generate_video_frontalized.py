import custom.tools as tools
import os
import numpy as np
import custom.faceExtractor as extractor
import cv2
import time
import pickle

def read_video(video_path):
  cap = cv2.VideoCapture(video_path)
  list_frames = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    list_frames.append(frame)
  cap.release()
  return list_frames


if __name__ == "__main__":
  
  only_align = False
  
  csv_path = os.path.join('partA','starting_point','subsample.csv')
  csv_array,_ = tools.get_array_from_csv(csv_path=csv_path)
  if only_align:
    video_saving_folder = os.path.join('partA','video','face_aligned')
    # pickle from partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl
  else:
    ref_landmarks = pickle.load(open(os.path.join('partA','video','mean_face_landmarks_per_subject','all_subjects_mean_landmarks.pkl'),'rb'))
    ref_landmarks= ref_landmarks['mean_facial_landmarks']
    video_saving_folder = os.path.join('partA','video','face_frontalized')
  video_folder_path = os.path.join('partA','video','video')
  # list_video_path = os.path.join(video_folder_path,csv_array[:,1],csv_array[:,-1])
  if not os.path.exists(video_saving_folder):
    os.makedirs(video_saving_folder)
  face_extractor = extractor.FaceExtractor()
  # reverse csv_array
  csv_array = csv_array[::-1]
  for user_id,video_sample in zip(csv_array[:,1],csv_array[:,-1]):
    start = time.time()
    video_path = os.path.join(video_folder_path,user_id,video_sample+'.mp4')
    list_frames = read_video(video_path)
    frame_aligned_list = []
    for frame in list_frames:
      if only_align:
        face = face_extractor.align_face(frame)
      else:
        face,_ = face_extractor.frontalize_img(frame=frame,
                                             ref_landmarks=ref_landmarks,
                                             frontalization_mode='SVD')
      if face is not None:
        frame_aligned_list.append(face)
      else:
        with open(os.path.join(video_saving_folder,'no_face.txt'),'a') as f:
          f.write('No face found in frame\n')
    tools.generate_video_from_list_frame(list_frame=frame_aligned_list,
                                       path_video_output=os.path.join(video_saving_folder,os.path.split(video_path)[-1]))
    print(f"Time taken: {time.time()-start:.2f} seconds")