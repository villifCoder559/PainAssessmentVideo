# crop_faces_parallel.py
import os
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

def worker_init(face_extractor_kwargs):
  """Runs in each worker process: create a module-level extractor."""
  global EXTRACTOR
  import custom.faceExtractor as faceExtractor
  EXTRACTOR = faceExtractor.FaceExtractor(**face_extractor_kwargs)

def process_single_video(video_path, root_dataset_path, face_crop_output_path, ref_landmarks, only_face_crop, overwrite=False):
  """
  Runs in worker process. Creates output path and runs frontalization + writes video.
  Returns (video_path, success_flag_or_error)
  """
  try:
    import custom.tools as tools

    # ensure output dir exists
    rel = os.path.relpath(video_path, root_dataset_path)
    output_path = os.path.join(face_crop_output_path, rel)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use the per-process EXTRACTOR set up in initializer
    dict_res = EXTRACTOR.frontalized_video(
      video_path=video_path,
      only_landmarks_crop=only_face_crop,
      ref_landmarks=ref_landmarks
    )

    # Basic checks: if no frames detected, return error
    if not any(dict_res.get('list_is_detected', [])):
      return (video_path, "No face detected in any frame")

    tools.generate_video_from_list_frame(
      list_frame=dict_res['list_frontalized_frame'],
      fps=dict_res['FPS'],
      path_video_output=output_path
    )
    return (video_path, True)
  except Exception as e:
    return (video_path, f"Exception: {repr(e)}")

def main():
  parser = argparse.ArgumentParser(description="Crop faces from CAER dataset videos (parallel).")
  parser.add_argument('--root_dataset_path', type=str, required=True)
  parser.add_argument('--face_crop_output_path', type=str, required=True)
  parser.add_argument('--from_', type=int, default=0)
  parser.add_argument('--to_', type=int, default=-1)
  parser.add_argument('--only_face_crop', type=int, choices=[0,1], default=1)
  parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers. Default: os.cpu_count() or fewer.')
  parser.add_argument('--face_detector_min_conf', type=float, default=0.1)
  args = parser.parse_args()

  root_dataset_path = args.root_dataset_path
  face_crop_output_path = args.face_crop_output_path

  # collect video paths
  list_video_path = []
  for root, dirs, files in os.walk(root_dataset_path):
    for file in files:
      if file.endswith(".avi"):
        list_video_path.append(os.path.join(root, file))
  list_video_path = list_video_path[args.from_: args.to_ if args.to_ != -1 else None]
  print(f"Total videos to process: {len(list_video_path)}. From index {args.from_} to {args.to_}")

  # Load reference landmarks once (small)
  ref_landmarks_path = "partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl"
  with open(ref_landmarks_path, 'rb') as f:
    ref_landmarks = pickle.load(f)
  ref_landmarks = ref_landmarks['mean_facial_landmarks']

  face_extractor_kwargs = dict(
    min_face_detection_confidence=args.face_detector_min_conf,
    min_face_presence_confidence=args.face_detector_min_conf,
    min_tracking_confidence=args.face_detector_min_conf
  )

  import os as _os
  max_workers = args.workers or max(1, _os.cpu_count() - 1)

  futures = []
  error_list = []
  count = 0

  with ProcessPoolExecutor(max_workers=max_workers,
                           initializer=worker_init,
                           initargs=(face_extractor_kwargs,)) as exe:
    for video_path in list_video_path:
      futures.append(exe.submit(process_single_video,
                                video_path,
                                root_dataset_path,
                                face_crop_output_path,
                                ref_landmarks,
                                args.only_face_crop == 1))

    for f in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
      video_path, res = f.result()
      if res is True:
        count += 1
      else:
        error_list.append((video_path, res))

  os.makedirs(face_crop_output_path, exist_ok=True)
  with open(os.path.join(face_crop_output_path, f"caer_face_crop_error_list_{args.from_}_{args.to_}.pkl"), 'wb') as fp:
    pickle.dump(error_list, fp)
  with open(os.path.join(face_crop_output_path, f"caer_face_crop_error_list_{args.from_}_{args.to_}.txt"), 'w') as fp:
    for video_path, error_msg in error_list:
      fp.write(f"{video_path}: {error_msg}\n")

  print(f"Processed {count} videos with {len(error_list)} errors.")

if __name__ == "__main__":
  main()
