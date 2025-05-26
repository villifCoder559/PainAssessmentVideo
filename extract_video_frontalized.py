import custom.faceExtractor as extractor
import custom.tools as tools
import os 
import pickle
import time 
import argparse
import numpy as np
from pathlib import Path
from custom.helper import GLOBAL_PATH
import numpy as np

def load_reference_landmarks(path):
  ref_landmarks = pickle.load(open(path, 'rb'))
  ref_landmarks = ref_landmarks['mean_facial_landmarks']
  return ref_landmarks

def main(generate_video,csv_path,path_folder_output,list_target_video,path_ref_landmarks,global_path,from_,to_,interpolation_mod_chunk,video_folder_path,align_before_front,log_error_path=None,only_oval=False):
  
  face_extractor = extractor.FaceExtractor(visionRunningMode='video')
  csv_list_video_path = np.array(tools.get_list_video_path_from_csv(csv_path=csv_path,
                                                                    video_folder_path=video_folder_path))
  root_video_path = os.path.split(os.path.split(csv_list_video_path[0])[0])[0]
  list_video_path = []
  # Init list_video_path with target video
  if list_target_video is not None:
    list_target_video = np.array([os.path.join(root_video_path,video.split('-')[0],video) for video in list_target_video])
    for video in list_target_video:
      # folder_sample = 
      mask = (csv_list_video_path == video)
      list_video_path.append(csv_list_video_path[mask][0])
  else:
      list_video_path = csv_list_video_path[from_:to_]
  print(f'\n csv path: {csv_path}\n')
  print(f'first video path: {list_video_path[0]}')
  print(f'last video path: {list_video_path[-1]}')
  print(f'Total video: {len(list_video_path)}\n')
  count = 0
  # '/media/villi/TOSHIBA EXT/orig_video/video/video/082414_m_64/082414_m_64-BL1-088.mp4'    
  # partA/video/video_frontalized/101514_w_36/101514_w_36-PA2-035.mp4
  # list_video_path = ['/media/villi/TOSHIBA EXT/orig_video/video/video/082814_w_46/082814_w_46-PA3-049.mp4']
  ref_landmarks = load_reference_landmarks(path=path_ref_landmarks)
  start = time.time()
  for video_path in list_video_path:
    print('Frontalizing video: ',video_path)
    if global_path:
      video_path = GLOBAL_PATH.get_global_path(video_path)
    try:
      # extra_landmark_smoothing = extractor.LandmarkSmoother(method='moving_average', window_size=5)
      dict_result_frontalization = face_extractor.frontalized_video(video_path=video_path,
                                                                    ref_landmarks=ref_landmarks,
                                                                    only_landmarks_crop=only_oval,
                                                                    align_before_front=align_before_front,
                                                                    interpolation_mod_chunk=interpolation_mod_chunk, # in order to get a multiple of the input number (ex: videoMae chunk is 16, so we need to set this to 16)
                                                                    extra_landmark_smoothing=None)
    except extractor.DetectionError as e:
      print(f'Error in {video_path}: {e}')
      with open(log_error_path,'a') as f:
        f.write(f'{video_path}: {e}\n')
        f.write(f'  list no detection: {e.list_no_detection_idx}\n')
      continue
    
    if generate_video:
      sample_id = os.path.split(video_path)[-1] # got sampleid.mp4
      folder_id = Path(video_path).parts[-2]
      out_path = os.path.join(path_folder_output,folder_id,sample_id)
      if global_path:
        out_path = GLOBAL_PATH.get_global_path(out_path)
      print(f'length of list frame: {len(dict_result_frontalization["list_frontalized_frame"])}')
      tools.generate_video_from_list_frame(list_frame=dict_result_frontalization['list_frontalized_frame'],
                                          path_video_output=out_path)
      print(f'Video frontalized: {out_path}')

    count += 1
    end = time.time()
    print(f'count: {count}/{len(list_video_path)}')
    print(f'Elapsed time: {int((end-start)/60/60)} h {int((end-start)/60%60)} m {int((end-start)%60)} s')
    predicted_time = ((end-start)/count*(len(list_video_path)-count))
    print(f'Time to end : {int(predicted_time/60/60)} h {int(predicted_time/60%60)} m {int(predicted_time%60)} s\n')
  
if __name__ == '__main__':
  def read_ltv(ltv):
    list_video = []
    with open(args.ltv[0],'r') as f:
      list_video = f.readlines()
    ltv = [video.strip() for video in list_video]
    return ltv
  
  def set_global_args(args):
    os.chdir(GLOBAL_PATH.NAS_PATH)
    args.csv = GLOBAL_PATH.get_global_path(args.csv)
    args.pfo = GLOBAL_PATH.get_global_path(args.pfo)
    args.prl = GLOBAL_PATH.get_global_path(args.prl)
    # args.log_er_p = GLOBAL_PATH.get_global_path(args.log_er_p)
    print(f'csv path: {args.csv}')
    print(f'path folder output: {args.pfo}')
    print(f'path ref landmarks: {args.prl}')
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--g_path',action='store_true', help='Add /equilibrium/fvilli/PainAssessmentVideo to all paths')
  parser.add_argument('--gv', action='store_true', help='Generate video')
  parser.add_argument('--csv', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--pfo', type=str, default=os.path.join('partA','video','video_frontalized'), help='Path to folder video output')
  parser.add_argument('--ltv', type=str, default=None,nargs='+', help='List of target video. Ex: video_name.txt ')
  parser.add_argument('--prl', type=str, default=os.path.join('partA','video','mean_face_landmarks_per_subject','all_subjects_mean_landmarks.pkl'), help='Path to reference landmarks')
  parser.add_argument('--from_', type=int, default=None, help='Frontalize video from index (included)')
  parser.add_argument('--to_', type=int, default=None, help='Frontalize video untill the index (excluded). Set None to get all video')
  # parser.add_argument('--log_er_p', type=str, default=os.path.join('partA','video','video_frontalized'), help='Path to log error file')
  parser.add_argument('--vfp', type=str, default=os.path.join('partA','video','video'), help='Path to video folder. Default to partA/video/video')
  parser.add_argument('--only_oval', action='store_true', help='Extract only face oval from the video')
  parser.add_argument('--interpolation_mod_chunk',nargs='*', default=None, help='mod can be spread_linearly or mirror_start_video, chunk is the chunk size in order to get a multiple number of frames according to chunk')
  parser.add_argument('--align_before_front', action='store_true', help='Align the face before frontalization')
  # sripts example: python3 extract_video_frontalized.py --g_path --gv --csv partA/starting_point/samples_exc_no_detection.csv --pfo partA/video/video_frontalized_new --prl partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl --from_ 0 --to_ 10
  args = parser.parse_args()
  if args.ltv and args.ltv[0].endswith('.txt'):
    args.ltv = read_ltv(args.ltv)
    
  if args.g_path:
    set_global_args(args)
  dict_args = vars(args)
  if dict_args['interpolation_mod_chunk'] and len(dict_args['interpolation_mod_chunk']) != 2:
    print(f'interpolation_mod_chunk must have 2 elements. mod can be spread_linearly or mirror_start_video, chunk is the chunk size in order to get a multiple number of frames according to chunk')
  for key, value in dict_args.items():
    print(f'{key}: {value}')
    
  main(generate_video=args.gv,
       csv_path=args.csv,
       path_folder_output=args.pfo,
       list_target_video=args.ltv,
       path_ref_landmarks=args.prl,
       global_path=args.g_path,
       from_=args.from_,
       to_=args.to_,
       log_error_path=args.pfo,
       only_oval=args.only_oval,
       align_before_front=args.align_before_front,
       interpolation_mod_chunk=[args.interpolation_mod_chunk[0],int(args.interpolation_mod_chunk[1])] if args.interpolation_mod_chunk else None,
       video_folder_path=args.vfp)
  
  