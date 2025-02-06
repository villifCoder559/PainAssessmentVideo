import custom.faceExtractor as extractor
import custom.tools as tools
import os 
import pickle
import time 
import argparse
import numpy as np
from pathlib import Path
from mediapipe.tasks.python import vision
from custom.helper import GLOBAL_PATH

def generate_path(path):
  return os.path.join(GLOBAL_PATH.NAS_PATH,path)

def load_reference_landmarks(path):
  ref_landmarks = pickle.load(open(path, 'rb'))
  ref_landmarks = ref_landmarks['mean_facial_landmarks']
  return ref_landmarks

def main(generate_video,stop_after,csv_path,path_folder_output,list_target_video,path_ref_landmarks,global_path,from_,to_,log_error_path=None):
  face_extractor = extractor.FaceExtractor()
  # csv_path = os.path.join('partA','starting_point','samples.csv') 
  # csv_array = tools.get_array_from_csv(csv_path=csv_path)
  csv_list_video_path = np.array(tools.get_list_video_path_from_csv(csv_path=csv_path))
  ref_landmarks = load_reference_landmarks(path=path_ref_landmarks)
  # ref_landmarks = face_extractor.get_flatten_landmarks(ref_landmarks)
  root_video_path = os.path.split(os.path.split(csv_list_video_path[0])[0])[0]
  list_video_path = []
  if list_target_video is not None:
    list_target_video = np.array([os.path.join(root_video_path,video.split('-')[0],video) for video in list_target_video])
    for video in list_target_video:
      # folder_sample = 
      mask = (csv_list_video_path == video)
      list_video_path.append(csv_list_video_path[mask][0])
  else:
      list_video_path = csv_list_video_path[from_:to_]
  print(f'\n \n \ncsv path: {csv_path}\n')
  print(f'first video path: {list_video_path[0]}')
  print(f'last video path: {list_video_path[-1]}')
  # frame_list = tools.get_list_frame_from_video_path(video_path=list_video_path[0])
  # 'partA/video/video/071309_w_21/071309_w_21-BL1-081.mp4'
  count = 0
  start = time.time()
  detector = vision.FaceLandmarker.create_from_options(face_extractor.options)
  for video_path in list_video_path:
    error = False
    if global_path:
      video_path = generate_path(video_path)
    list_frontalized_frames = []
    frame_list = tools.get_list_frame_from_video_path(video_path=video_path)
    print(f'\nFrontalizing {video_path}...')
    for k,frame in enumerate(frame_list):
      frame_frontalized,_ = face_extractor.frontalize_img(frame=frame,
                                                          frontalization_mode='SVD',
                                                          ref_landmarks=ref_landmarks,
                                                          time_logs=True,
                                                          v2=True,
                                                          detector=detector)
      if frame_frontalized is None:
        if not os.path.exists(log_error_path):
          os.makedirs(log_error_path)
        with open(os.path.join(log_error_path,'error_frontalization.txt'),'a') as f:
          f.write(f'{video_path} frame {k} \n')
          print(f'Log error in error_frontalization.txt')
        error = True
        break
      if (k+1) % 40 == 0:
        print(f'  {k+1}/{len(frame_list)} frames frontalized')
      # print(f'Landmarks left eye position:\n {rotated_landmarks[face_extractor.LEFT_CORNER_EYE_INDEXES][:2]*frame_frontalized.shape}')
      list_frontalized_frames.append(frame_frontalized)
    # for k,v in time_dict.items():
    #   print(f'  {k}: {v:.2f} s')
    # print(f' Time to frontalized {len(frame_list)} frames: {end-start:.2f} s')
    # face_extractor._reset_total_time()
    if generate_video and not error:
      sample_id = os.path.split(video_path)[-1] # got sampleid.mp4
      folder_id = Path(video_path).parts[-2]
      out_path = os.path.join(path_folder_output,folder_id,sample_id)
      if global_path:
        out_path = generate_path(out_path)
      tools.generate_video_from_list_frame(list_frame=list_frontalized_frames,
                                          path_video_output=out_path)
    else:
      print(f'Error in {video_path}')
    count += 1
    end = time.time()
    print(f'count: {count}/{len(list_video_path)}')
    print(f'Elapsed time: {int((end-start)/60/60)} h {int((end-start)/60%60)} m {int((end-start)%60)} s')
    predicted_time = ((end-start)/count*(len(list_video_path)-count))
    print(f'Finish time : {int(predicted_time/60/60)} h {int(predicted_time/60%60)} m {int(predicted_time%60)} s\n')
    if count == stop_after:
      break
    

  
if __name__ == '__main__':
  # 'partA','video','features','samples_16_frontalized_with_delta','video'
  parser = argparse.ArgumentParser()
  parser.add_argument('--g_path',action='store_true', help='Add /equilibrium/fvilli/PainAssessmentVideo to all paths')
  parser.add_argument('--gv', action='store_true', help='Generate video')
  parser.add_argument('--stop', type=int,default=-1, help='Stop after n videos')
  parser.add_argument('--csv', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--pfo', type=str, default=os.path.join('partA','video','video_frontalized'), help='Path to folder video output')
  parser.add_argument('--ltv', type=str, default=None,nargs='+', help='List of target video. Ex: video_name.ext ')
  parser.add_argument('--prl', type=str, default=os.path.join('partA','video','mean_face_landmarks_per_subject','all_subjects_mean_landmarks.pkl'), help='Path to reference landmarks')
  parser.add_argument('--from_', type=int, default=None, help='Frontalize video from index (included)')
  parser.add_argument('--to_', type=int, default=None, help='Frontalize video untill the index (excluded). Set None to get all video')
  parser.add_argument('--log_er_p', type=str, default=os.path.join('partA','video','video_frontalized'), help='Path to log error file')
  args = parser.parse_args()
  if args.ltv[0].endswith('.txt'):
    list_video = []
    with open(args.ltv[0],'r') as f:
      list_video = f.readlines()
    args.ltv = [video.strip() for video in list_video]
  if args.g_path:
    os.chdir('/equilibrium/fvilli/PainAssessmentVideo')
    args.csv = generate_path(args.csv)
    args.pfo = generate_path(args.pfo)
    args.prl = generate_path(args.prl)
    args.log_er_p = generate_path(args.log_er_p)
    print(f'csv path: {args.csv}')
    print(f'path folder output: {args.pfo}')
    print(f'path ref landmarks: {args.prl}')
  main(generate_video=args.gv,
       stop_after=args.stop,
       csv_path=args.csv,
       path_folder_output=args.pfo,
       list_target_video=args.ltv,
       path_ref_landmarks=args.prl,
       global_path=args.g_path,
       from_=args.from_,
       to_=args.to_,
       log_error_path=args.log_er_p)
  
  