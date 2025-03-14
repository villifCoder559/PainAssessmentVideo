import cv2
import numpy as np
import os
import custom.tools as tools
import custom.faceExtractor as extractor
from pathlib import Path
import time
import dlib

def print_log_error_details(list_path_video,list_no_det_frame,path_save_log_annotation=None):
  
  def _merge_dict_list(dict_list,key_to_oreder='nr_frames'):
    name_dict = list(dict_list[0].keys())
    if key_to_oreder not in name_dict:
      raise ValueError(f'key_to_oreder: {key_to_oreder} not in name_dict: {name_dict}')
    idx_key = name_dict.index(key_to_oreder)
    list_array = []
    for key in name_dict:
      tmp_key = np.array([el[key] for el in dict_list])
      list_array.append(tmp_key)
    
    sorted_idx = np.argsort(list_array[idx_key])[::-1]
    zipped = zip(*[list_array[idx][sorted_idx] for idx in range(len(list_array))])
    list_merged = np.array([{name_dict[idx]:el for idx,el in enumerate(el_zipped)} for el_zipped in zipped])
    # list_merged = np.array([{f'{name_dict[0]}':sbj,f'{name_dict[1]}':nr_sample_video} for sbj,nr_sample_video in zip(tmp_name[sorted_idx],
    #                                                                                                                 tmp_val[sorted_idx])])
    return list_merged
  
  nr_frame_no_det = len(list_path_video)
  # partA/video/video/{SUBJECT}/{SAMPLE}.mp4
  list_subjects = np.array([Path(path).parts[3] for path in list_path_video])
  list_sample_video = np.array([Path(path).parts[4] for path in list_path_video])
  list_unique_subjects = np.unique(list_subjects, return_index=False)
  list_unique_sample_video = np.unique(list_sample_video, return_index=False)
  list_subject_video = []
  list_no_det_frame = np.array(list_no_det_frame,dtype=int)
  nr_video = 0
  for sbj in list_unique_subjects:
    mask = list_subjects == sbj
    list_sample_video_sbj = list_sample_video[mask]
    list_frames_sbj = list_no_det_frame[mask]
    unique_list_sample_video_sbj = np.unique(list_sample_video_sbj, return_index=False)
    list_subject_video.append({
      'subject': sbj,
      'nr_sample_video': len(unique_list_sample_video_sbj),
      'nr_frames': len(list_frames_sbj)
    })
    # print(f' {sbj}: {len(unique_list_sample_video_sbj)}')
    nr_video += len(unique_list_sample_video_sbj)
  list_sample_frames = []
  for sample in list_unique_sample_video:
    mask = list_sample_video == sample
    list_sample_frames.append({
      'sample': sample,
      'nr_frames': len(list_sample_video[mask]),
    })
    # print(f' {sample}: {len(list_sample_video[mask])}')
    # nr_video += len(list_sample_video[mask]) 
    
  
  print(f'nr_frame_no_det: {nr_frame_no_det}')
  print(f'nr_unique_subjects: {len(list_unique_subjects)}')
  print(f'nr_video {nr_video}')
  print(f'For each subject, the number of sample video that have no detected face in at least one frame:')
  list_subject_video = _merge_dict_list(list_subject_video)
  print(list_subject_video)
  list_sample_frames = _merge_dict_list(list_sample_frames)
  print(f'For each sample video, the number of frames that have no detected face:')
  print(list_sample_frames)
  if path_save_log_annotation is not None:
    with open(path_save_log_annotation,'w') as file:
      file.write(f'nr_frame_no_det: {nr_frame_no_det}\n')
      file.write(f'nr_unique_subjects: {len(list_unique_subjects)}\n')
      file.write(f'nr_video {nr_video}\n')
      file.write(f'For each subject, the number of sample video that have no detected face in at least one frame:\n')
      for el in list_subject_video:
        file.write(f'{el}\n')
      file.write(f'For each sample video, the number of frames that have no detected face:\n')
      for el in list_sample_frames:
        file.write(f'{el}\n')
    print(f'log annotation saved in: {path_save_log_annotation}')

def _elaborate_img(face_extractor,img,timestamp,apply_align=False,apply_landmarks_detection=False):
  # Suppose that there is only one face in the image
  annotated_img = img.copy()
  annotations = {
    'align':None,
    'detection':None
  }
  if apply_align:
    annotated_img = face_extractor.align_face(annotated_img)
    if annotated_img is None:
      annotations['align'] = 'No detection'
      annotated_img = img.copy()
    else:
      annotations['align'] = 'Detected'
  if apply_landmarks_detection:
    detection_result = face_extractor.extract_facial_landmarks([(annotated_img,timestamp)])[0].face_landmarks # 0 because there is one image
    if len(detection_result)==0:
      annotations['detection'] = 'No detection'
      annotated_img = img.copy()
    else:
      annotations['detection'] = 'Detected'
      annotated_img,_ = face_extractor.plot_landmarks(image=annotated_img,
                                                    landmarks=detection_result[0],
                                                    connections=face_extractor.FACE_TESSELATION)
  return annotated_img,annotations
 
def write_video_text_annotations(annotations,annotated_img,count_annotation):
  if annotations['align'] is not None:
    cv2.putText(annotated_img,f'align_anno: '+ annotations['align'],(25,145),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    if annotations['align'] == 'Detected':
      count_annotation['align_recognized'] += 1
    else:
      count_annotation['align_not_recognized'] += 1
  if annotations['detection'] is not None:
    cv2.putText(annotated_img,f'detec_anno: '+ annotations['detection'],(25,185),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    if annotations['detection'] == 'Detected':
      count_annotation['detection_recognized'] += 1
    else:
      count_annotation['detection_not_recognized'] += 1
  if (annotations['align'] == 'Detected' and annotations['detection'] == 'Detected') or (annotations['align'] == 'No detection' and annotations['detection'] == 'No detection'):
    count_annotation['intersection_same_output'] += 1
  else:
    count_annotation['intersection_different_output'] += 1

   
def generate_unique_video(list_path_video,list_nr_frame,path_save_video,face_extractor,path_save_log_annotation):
  count_annotation = {
    'align_recognized':0,
    'align_not_recognized':0,
    'detection_recognized':0,
    'detection_not_recognized':0,
    'intersection_same_output':0,
    'intersection_different_output':0,
    'dlib_detection':0,
    'dlib_fail_detection':0
  }
  list_no_detection = []
  list_path_video = np.array(list_path_video)
  unique_list_path_video = np.unique(list_path_video,return_index=False)
  unique_list_frame = []
  list_nr_frame = np.array(list_nr_frame,dtype=int)
  for path in unique_list_path_video:
    mask = list_path_video == path
    list_frame_video = list_nr_frame[mask]
    unique_list_frame.append(list_frame_video)

  out = None
  path_save_video = os.path.join(path_save_video,'video.mp4')
  count_video = 1
  start = time.time()
  out_fps = 4
  list_start_video_sbj_= []
  count_frame = 0
  # dlib_detector,dlib_predictor = create_dlib_detector()
  with open(path_save_log_annotation,'w') as file:
    file.write(f'{face_extractor.config}\n')
  for path,list_nr_frame in zip(unique_list_path_video,unique_list_frame):
    subject = Path(path).parts[3]
    sample = Path(path).parts[4]
    cap = cv2.VideoCapture(path)
    video_frame_width,video_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out is None:
      print(f'out is None')
      out = cv2.VideoWriter(path_save_video,cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (video_frame_width,video_frame_height))
    if cap.isOpened() == False:
      print(f'Error: {path} not opened')
    else:
      start_video_time_in_video = {
        'path':path,
        'timeline_h_m_s':f'{int(count_frame/out_fps/60/60)}:{int(count_frame/out_fps/60)%60}:{int(count_frame/out_fps)%60}',
      }  
      list_start_video_sbj_.append(start_video_time_in_video)
      print(f'path: {path}')
      for frame in list_nr_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not ret:
          print(f'Error: {path} not read')
        else:
          annotated_img, annotations = _elaborate_img(img=img,
                                                      face_extractor=face_extractor,
                                                      timestamp=frame,
                                                      apply_align=True,
                                                      apply_landmarks_detection=True)
          # dlib_results = detect_landmarks_dlib(image=img,detector=dlib_detector)
          # if len(dlib_results) == 1:
          #   count_annotation['dlib_detection'] += 1
          # else:
          #   count_annotation['dlib_fail_detection'] += 1
          if annotated_img.shape != (video_frame_height,video_frame_width,3):
            print(f'Annotated_img shape{annotated_img.shape} not equal to video shape {(video_frame_height,video_frame_width,3)}')
            whole_img = np.zeros((video_frame_height,video_frame_width,3),dtype=np.uint8)+255
            annotated_img = cv2.resize(annotated_img,(annotated_img.shape[1]*2,annotated_img.shape[0]*2))
            left_corner_centered = (int((video_frame_width-annotated_img.shape[1])/2),int((video_frame_height-annotated_img.shape[0])/2))
            right_corner_centered = (left_corner_centered[0]+annotated_img.shape[1],left_corner_centered[1]+annotated_img.shape[0])
            whole_img[left_corner_centered[1]:right_corner_centered[1],left_corner_centered[0]:right_corner_centered[0]] = annotated_img
            annotated_img = whole_img
          if annotations['align'] == 'No detection':
            list_no_detection.append({
              'type':'align',
              'sample':sample,
              'timestamp_h_m_s':f'{int(frame/out_fps/60/60)}:{int(frame/out_fps/60)%60}:{int(frame/out_fps)%60}'
            })
          if annotations['detection'] == 'No detection':
            list_no_detection.append({
              'type':'detection',
              'sample':sample,
              'timestamp_h_m_s':f'{int(frame/out_fps/60/60)}:{int(frame/out_fps/60)%60}:{int(frame/out_fps)%60}'
            })
          cv2.putText(annotated_img,f'subject  : {subject}',(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
          cv2.putText(annotated_img,f'sample   : {sample}',(25,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
          cv2.putText(annotated_img,f'frame    : {frame}',(25,105),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
          write_video_text_annotations(annotations=annotations,
                                       annotated_img=annotated_img,
                                       count_annotation=count_annotation)
          # cv2.putText(annotated_img,f'dlib_detection: {"Detection" if len(dlib_results)==1 else "No detection"}',(25,225),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
          # get the time in the out video when the frame is read
          count_frame += 1
          annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
          out.write(annotated_img)
      
      end = time.time()
      print('-'*100)
      print(f'video: {subject}_{sample}')
      print(f'Analyzed {count_video}/{len(unique_list_path_video)} video')
      print(f'Elapsed time: {int((end-start)/60)} m {int((end-start)%60)} s')
      predicted_time = (end-start)/count_video*len(unique_list_path_video)
      print(f'Predicted time: {int(predicted_time/60)%60} m {int(predicted_time%60)} s')
      print('-'*100)
      count_video += 1
    cap.release()
  out.release()
  folder_annotation = os.path.split(path_save_log_annotation)[0]
  # if path_save_log_annotation is not None:
  with open(os.path.join(folder_annotation,'annotation.txt'),'w') as file:
    for el in list_no_detection:
      for k,v in el.items():
        file.write(f'{k}: {v}\n')
      file.write('\n')
  with open(os.path.join(folder_annotation,'timeline.txt'),'w') as file:
    for el in list_start_video_sbj_:
      file.write(f'{el}\n')
    print(f'Timeline saved in: {os.path.join(folder_annotation,"timeline.txt")}')
  
  with open(os.path.join(folder_annotation,'count_detection.txt'),'w') as file:
    print(f'Count of detection for different functions using MediaPipe lib')
    for k,v in count_annotation.items():
      file.write(f'{k}: {v}\n')
    file.write(f'total frame: {count_frame}\n')
    print(f'Count detection saved in: {os.path.join(folder_annotation,"count_detection.txt")}')
  
  print(f'Video saved in: {path_save_video}')
  
def detect_landmarks_dlib(image,detector):
  # image = cv2.imread("apple.jpeg")
  image = cv2.imread("output.png")
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)
  return faces

def create_dlib_detector(predictor_path= os.path.join('landmark_model','dlib_shape_predictor_68_face_landmarks.dat')):
  predictor_path = predictor_path
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)
  return detector,predictor

if __name__ == '__main__':
  
  path_log_not_detected_file = os.path.join('partA','video','mean_face_landmarks_per_subject','no_detection_log.txt')
  # read path_log_not_detected_file
  unique_list_frame = []
  with open(path_log_not_detected_file, 'r') as file:
    lines = file.readlines()
  list_path_video = [line.strip().split(',')[0] for line in lines]
  list_no_det_frame = [line.strip().split(',')[1] for line in lines]
  
  path_save_video = os.path.join('partA','video','mean_face_landmarks_per_subject','video')
  logs_file_name = 'logs'+str(int(time.time()))
  path_save_log_annotation = os.path.join('partA','video','mean_face_landmarks_per_subject','video',logs_file_name)
  
  if not os.path.exists(path_save_video):
    os.makedirs(path_save_video)
  if not os.path.exists(path_save_log_annotation):
    os.makedirs(path_save_log_annotation)
    
  path_save_log_annotation_video = os.path.join(path_save_log_annotation,'video_log_annotation.txt')
  
  print_log_error_details(list_path_video=list_path_video,
                          list_no_det_frame=list_no_det_frame,
                          path_save_log_annotation = os.path.join(path_save_log_annotation,'log_error_details.txt')
                          )
  face_extractor = extractor.FaceExtractor(
    visionRunningMode='video',
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    model_path= os.path.join('landmark_model','face_landmarker_v2_with_blendshapes.task'),
  )
  generate_unique_video(list_path_video=list_path_video,
                        list_nr_frame=list_no_det_frame,
                        face_extractor=face_extractor,
                        path_save_video=path_save_video,
                        path_save_log_annotation=path_save_log_annotation_video)
    