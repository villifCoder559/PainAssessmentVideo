import numpy as np
import cv2
import mediapipe as mp
import time
# import custom.tools as tools
import os
import pandas as pd
import copy
import pickle

def get_roll_pitch_yaw_from_video(video_path,mp_face_mesh,face_mesh,show_video=False,error_log_folder=os.path.join("partA","video","roll_pitch_yaw_per_subject")):
  if show_video:
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)
  # print(f'video_path: {video_path}')
  # video_path = 'partA/video/video/112016_m_25/112016_m_25-BL1-081.mp4'
  cap = cv2.VideoCapture(video_path)
  # cap = cv2.VideoReader
  dict_roll = {}
  dict_pitch = {}
  dict_yaw = {}
  if cap.isOpened() == False:
    print(f'Error opening video file: {video_path}')
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      if show_video:
        cv2.destroyAllWindows()
      break
    # print(f'success: {success}')
    start = time.time()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #flipped for selfie view
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    img_h , img_w, img_c = image.shape
    face_2d = []
    face_3d = []
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
          # 1 is nose, 33 right eye ("left corner"), 263 left eye ("right corner"), 61 mouth corner left,  291 mouth corner right, 199 chin
          # 133 right eye("right corner"), 362 left eye ("left corner")
          if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199 or idx == 133 or idx == 362:
            if idx ==1:
              nose_2d = (lm.x * img_w,lm.y * img_h)
              # nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
            x,y = int(lm.x * img_w),int(lm.y * img_h)
            face_2d.append([x,y])
            face_3d.append(([x,y,lm.z]))
        #Get 2d Coord
        face_2d = np.array(face_2d,dtype=np.float64)
        face_3d = np.array(face_3d,dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length,0,img_h/2],
                              [0,focal_length,img_w/2],
                              [0,0,1]])
        distortion_matrix = np.zeros((4,1),dtype=np.float64)
        success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)
        #getting rotational of face
        rmat,jac = cv2.Rodrigues(rotation_vec)
        angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)
        x = angles[0] * 360 # roll
        y = angles[1] * 360 # pitch
        z = angles[2] * 360 # yaw
        dict_roll[int(x)] = dict_roll.get(int(x),0) + 1
        dict_pitch[int(y)] = dict_pitch.get(int(y),0) + 1
        dict_yaw[int(z)] = dict_yaw.get(int(z),0) + 1
        p1 = (int(nose_2d[0]),int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))
        # print(f'x: {x}, y: {y}, z: {z}')
        if show_video:
          # here based on axis rot angle is calculated
          if y < -10:
              text="Looking Left"
          elif y > 10:
              text="Looking Right"
          elif x < -10:
              text="Looking Down"
          elif x > 10:
              text="Looking Up"
          else:
              text="Forward"
          # nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)
          cv2.line(image,p1,p2,(255,0,0),3)
          cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
          cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
          cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
          cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      end = time.time()
      totalTime = end-start
      
      if show_video:
        fps = 1/totalTime
        cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)
    else:
      with open(os.path.join(error_log_folder,'error_log.txt'),'a') as f:
        f.write(f'No face detected in {video_path}\n')
    if show_video:
      cv2.imshow('Head Pose Detection',image)
      if cv2.waitKey(5) & 0xFF ==27:
          break
    
  cap.release()
  dict_angles = {
    'roll':dict_roll,
    'pitch':dict_pitch,
    'yaw':dict_yaw
  }
  return dict_angles

def save_results(dict_results,root_folder_path):
  if not os.path.exists(root_folder_path):
    os.makedirs(root_folder_path)
  for sbj_name in dict_results.keys():
    subject_folder_path = os.path.join(root_folder_path,sbj_name)
    if not os.path.exists(subject_folder_path):
      os.makedirs(os.path.join(subject_folder_path))
    with open(os.path.join(subject_folder_path,f'log_{sbj_name}.txt'),'w') as f:
      # f.write(f'{sbj_name},')
      for key in dict_results[sbj_name].keys():
        f.write(f'{key}:\n')
        for angle in dict_results[sbj_name][key].keys():
          f.write(f' {angle}:{dict_results[sbj_name][key][angle]}\n')
    pickle.dump(dict_results[sbj_name],open(os.path.join(subject_folder_path,f'dict_{sbj_name}.pkl'),'wb'))
    print(f'{sbj_name} results saved in {subject_folder_path}')
  pickle.dump(dict_results,open(os.path.join(root_folder_path,'dict_results_all_subjects.pkl'),'wb'))
  print(f'All results saved in {root_folder_path}')  
    
def get_array_from_csv(csv_path):
  """
  Reads a CSV file, converts it to a NumPy array, and processes each entry by splitting
  the first column using a tab delimiter. The processed entries are then stacked into
  a single NumPy array.

  Args:
    csv_path (str): The file path to the CSV file.

  Returns:
    np.ndarray: A NumPy array containing the processed entries from the CSV file.\n
                (BIOVID cols-> subject_id, subject_name, class_id, class_name, sample_id, sample_name)
  """
  csv_array = pd.read_csv(csv_path)  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  cols_array = csv_array.columns.to_numpy()[0].split('\t')
  csv_array = csv_array.to_numpy()
  list_samples = []
  for entry in csv_array:
    tmp = entry[0].split("\t")
    list_samples.append(tmp)
  return np.stack(list_samples),cols_array

if __name__ == '__main__':
  # video_path = 'partA/video/video/112016_m_25/112016_m_25-BL1-081.mp4'
  csv_path = os.path.join('partA','starting_point','samples_exc_no_detection.csv')
  # root_video_path = os.path.join('partA','video','video')
  root_video_path = "/media/villi/TOSHIBA EXT/orig_video/video/video"
  csv,_ = get_array_from_csv(csv_path)
  dict_results = {}
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
  list_subject_name = np.unique(csv[:,1])
  list_sample = csv[:,5]
  for sbj_name in list_subject_name:
    dict_roll = {}
    dict_pitch = {}
    dict_yaw = {}
    start = time.time()
    count = 0
    sample_subject = csv[:,1] == sbj_name
    sample_list = csv[:,5][sample_subject]
    for sample_name in sample_list:
      count += 1
      video_path = os.path.join(root_video_path,sbj_name,sample_name)    
      dict_angles = get_roll_pitch_yaw_from_video(video_path=video_path+'.mp4',
                                                  mp_face_mesh=mp_face_mesh,
                                                  face_mesh=face_mesh,
                                                  show_video=False)
      dict_results[sample_name] = copy.deepcopy(dict_angles)
      # for k,v in dict_angles.items():
      #   for angle in v.keys():
      #     if k == 'roll':
      #       dict_roll[angle] = dict_roll.get(angle,0) + v[angle]
      #     elif k == 'pitch':
      #       dict_pitch[angle] = dict_pitch.get(angle,0) + v[angle]
      #     else:
      #       dict_yaw[angle] = dict_yaw.get(angle,0) + v[angle]
      if count % 10 == 0:
        print(f'{sbj_name} done {count} samples')
    end = time.time()
    
    print(f'{sbj_name} done in {end-start} seconds')
    # print(f'dict roll: {dict_roll}')
    # print(f'dict pitch: {dict_pitch}')
    # print(f'dict yaw: {dict_yaw}')
    # dict_results[sbj_name] = {'roll':copy.deepcopy(dict_roll),'pitch':copy.deepcopy(dict_pitch),'yaw':copy.deepcopy(dict_yaw)}

  print(f'current working directory: {os.getcwd()}')
  with open(os.path.join('partA','video','roll_pitch_yaw_per_subject_all'),'wb') as f:
    pickle.dump(dict_results,f)
  print(f'All results saved in partA/video/roll_pitch_yaw_per_subject_all')
  face_mesh.close()
  # save_results(dict_results,os.path.join('partA','video','roll_pitch_yaw_per_subject_all'))
  