#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import tqdm
import multiprocessing as mp
from multiprocessing import Manager

import custom.faceExtractor as extractor
import custom.tools as tools
from custom.helper import GLOBAL_PATH

# Global variables initialized in worker initializer
face_extractor = None
worker_config = None
log_lock = None

def load_reference_landmarks(path):
  with open(path, 'rb') as f:
    ref_landmarks = pickle.load(f)
  return ref_landmarks['mean_facial_landmarks']

def worker_init(ref_landmarks, cfg, shared_dict):
  """
  Called once per worker process. Creates a FaceExtractor for worker.
  """
  global face_extractor, worker_config, _dict
  _dict = shared_dict
  worker_config = cfg
  face_extractor = extractor.FaceExtractor(
    visionRunningMode='video',
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3
  )
  worker_config['ref_landmarks'] = ref_landmarks

def process_one_video(video_path):
  """
  Process a single video. Runs inside worker process.
  """
  global face_extractor, worker_config, _dict
  out = {'video': video_path, 'ok': False, 'error': None}

  try:
    original_video_path = video_path
    if worker_config.get('global_path'):
      video_path = GLOBAL_PATH.get_global_path(video_path)

    sample_id = os.path.split(video_path)[-1]
    if 'caer' not in video_path.lower():
      folder_id = Path(video_path).parts[-2]
    else:
      folder_id = os.path.join(*Path(video_path).parts[-3:-1])
    out_path = os.path.join(worker_config['path_folder_output'], folder_id, sample_id)
    if worker_config.get('global_path'):
      out_path = GLOBAL_PATH.get_global_path(out_path)

    dict_result = face_extractor.frontalized_video(
      video_path=original_video_path,
      ref_landmarks=worker_config['ref_landmarks'],
      only_landmarks_crop=worker_config['only_oval'],
      align_before_front=worker_config['align_before_front'],
      interpolation_mod_chunk=worker_config['interpolation_mod_chunk'],
      extra_landmark_smoothing=None
    )

    if worker_config.get('generate_video'):
      tools.generate_video_from_list_frame(
        list_frame=dict_result['list_frontalized_frame'],
        fps=dict_result['FPS'],
        path_video_output=out_path
      )

    out['ok'] = True
    return out

  except extractor.DetectionError as e:
    msg = f'{video_path}: {e}'
    _dict[f'error_{os.getpid()}'] = e.list_no_detection_idx
    out['error'] = str(e)
    return out
  
  except Exception as e:
    _dict[f'error_{os.getpid()}'] = str(e)
    out['error'] = repr(e)
    return out

def prepare_video_list(csv_path, video_folder_path, list_target_video, from_, to_):
  
  csv_list_video_path = np.array(
    tools.get_list_video_path_from_csv(csv_path=csv_path,
                                       video_folder_path=video_folder_path,
                                       extension='.avi' if 'caer' in csv_path.lower() else '.mp4')
  )
  root_video_path = os.path.split(os.path.split(csv_list_video_path[0])[0])[0]

  if list_target_video is not None:
    list_target_video = np.array([os.path.join(root_video_path, v.split('-')[0], v)
                                  for v in list_target_video])
    list_video_path = []
    for video in list_target_video:
      mask = (csv_list_video_path == video)
      if np.any(mask):
        list_video_path.append(csv_list_video_path[mask][0])
  else:
    list_video_path = csv_list_video_path[from_:to_]

  return list_video_path

def main(generate_video, csv_path, path_folder_output, list_target_video,
         path_ref_landmarks, global_path, from_, to_, interpolation_mod_chunk,
         video_folder_path, align_before_front, log_error_path=None,
         only_oval=False, workers=4):

  csv_path = os.path.expanduser(csv_path)
  path_folder_output = os.path.expanduser(path_folder_output)
  if log_error_path is None:
    log_error_path = os.path.join(path_folder_output, 'frontalize_errors.log')

  ref_landmarks = load_reference_landmarks(path_ref_landmarks)
  list_video_path = prepare_video_list(csv_path, video_folder_path,
                                       list_target_video, from_, to_)

  if len(list_video_path) == 0:
    print("No videos to process. Exiting.")
    return

  print(f'\n csv path: {csv_path}\n')
  print(f'first video path: {list_video_path[0]}')
  print(f'last video path: {list_video_path[-1]}')
  print(f'Total video: {len(list_video_path)}\n')

  cfg = {
    'path_folder_output': path_folder_output,
    'generate_video': generate_video,
    'log_error_path': log_error_path,
    'only_oval': only_oval,
    'align_before_front': align_before_front,
    'interpolation_mod_chunk': interpolation_mod_chunk,
    'global_path': global_path
  }

  Path(path_folder_output).mkdir(parents=True, exist_ok=True)

  manager = Manager()
  shared_dict = manager.dict()

  pool = mp.Pool(processes=workers,
                 initializer=worker_init,
                 initargs=(ref_landmarks, cfg, shared_dict))

  results_iter = pool.imap_unordered(process_one_video, list_video_path)

  success = 0
  errors = 0
  with tqdm.tqdm(total=len(list_video_path),
                 desc='Frontalizing videos',
                 unit='video') as pbar:
    for res in results_iter:
      pbar.update(1)
      if res.get('ok'):
        success += 1
      else:
        errors += 1

  pool.close()
  pool.join()
  with open(os.path.join(log_error_path,f'frontalize_errors_{from_}_{to_}.log'), 'w') as f:
    for key in shared_dict.keys():
      f.write(f'{key}: {shared_dict[key]}\n')
  print(f'\nDone. Success: {success}, Errors: {errors}')

if __name__ == '__main__':
  def read_ltv(ltvfile):
    with open(ltvfile, 'r') as f:
      return [line.strip() for line in f.readlines()]

  parser = argparse.ArgumentParser()
  parser.add_argument('--g_path', action='store_true', help='Use GLOBAL_PATH mappings')
  parser.add_argument('--gv', action='store_true', help='Generate video')
  parser.add_argument('--vfp', type=str, default=os.path.join('partA','video','video'),
                      help='Path to video folder')
  parser.add_argument('--csv', type=str,
                      default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'),
                      help='Path to csv file')
  parser.add_argument('--pfo', type=str,
                      default=os.path.join('partA','video','video_frontalized'),
                      help='Path to folder video output')
  parser.add_argument('--ltv', type=str, default=None, nargs='+',
                      help='List of target video. Ex: video_name.txt ')
  parser.add_argument('--prl', type=str,
                      default=os.path.join('partA','video','mean_face_landmarks_per_subject','all_subjects_mean_landmarks.pkl'),
                      help='Path to reference landmarks')
  parser.add_argument('--from_', type=int, default=None,
                      help='From index (included)')
  parser.add_argument('--to_', type=int, default=None,
                      help='To index (excluded)')
  parser.add_argument('--only_oval', action='store_true',
                      help='Extract only face oval from the video')
  parser.add_argument('--interpolation_mod_chunk', nargs='*', default=None,
                      help='[mod, chunk]')
  parser.add_argument('--align_before_front', action='store_true',
                      help='Align the face before frontalization')
  parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker processes for parallel execution')
  args = parser.parse_args()

  if args.ltv and args.ltv[0].endswith('.txt'):
    args.ltv = read_ltv(args.ltv[0])

  if args.g_path:
    os.chdir(GLOBAL_PATH.NAS_PATH)
    args.csv = GLOBAL_PATH.get_global_path(args.csv)
    args.pfo = GLOBAL_PATH.get_global_path(args.pfo)
    args.prl = GLOBAL_PATH.get_global_path(args.prl)
    print(f'csv path: {args.csv}')
    print(f'path folder output: {args.pfo}')
    print(f'path ref landmarks: {args.prl}')

  if args.interpolation_mod_chunk and len(args.interpolation_mod_chunk) != 2:
    raise SystemExit('interpolation_mod_chunk must have 2 elements: mod and chunk')

  interp = None
  if args.interpolation_mod_chunk:
    interp = [args.interpolation_mod_chunk[0], int(args.interpolation_mod_chunk[1])]

  print("Running with args:")
  for k, v in vars(args).items():
    print(f'  {k}: {v}')

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
       interpolation_mod_chunk=interp,
       video_folder_path=args.vfp,
       workers=args.workers)
