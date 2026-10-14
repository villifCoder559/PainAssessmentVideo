"""
Average the MediaPipe face landmarks of every BioVid subject into the reference geometry
used by extract_video_frontalized.py (--prl).

Writes <out_folder>/<subject_name>.pkl per subject and <out_folder>/all_subjects_mean_landmarks.pkl,
the mean over subjects. Subjects whose pickle already exists are read back, not recomputed.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import custom.faceExtractor as extractor
import custom.tools as tools


def subject_landmarks(job):
  """
  Mean landmarks of one subject, computed or read back from its pickle.

  Args:
    job: (subject_name, video_root, out_folder).

  Returns:
    (subject_name, mean landmarks of shape (478, 3)).
  """
  subject_name, video_root, out_folder = job
  pkl_path = os.path.join(out_folder, subject_name + '.pkl')
  if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
      return subject_name, pickle.load(f)['mean_facial_landmarks']
  folder_path = os.path.join(video_root, subject_name)
  list_video_path = [os.path.join(folder_path, video) for video in os.listdir(folder_path)]
  mean_facial_landmarks, count_frame = extractor.FaceExtractor().get_mean_facial_landmarks(
    list_video_path=list_video_path, align=True, numpy_view=True,
    no_detection_log=os.path.join(out_folder, 'no_detection_log.txt'))
  dict_subject = {'subject_name': subject_name,
                  'mean_facial_landmarks': mean_facial_landmarks,
                  'count_frame': count_frame}
  with open(pkl_path, 'wb') as f:
    pickle.dump(dict_subject, f)
  return subject_name, mean_facial_landmarks


def main(video_root, csv_path, out_folder, workers=1):
  csv_array, cols = tools.get_array_from_csv(csv_path=csv_path)
  list_subject_name = np.unique(csv_array[:, 1])
  os.makedirs(out_folder, exist_ok=True)
  start = time.time()
  per_subject = {}
  jobs = [(subject_name, video_root, out_folder) for subject_name in list_subject_name]
  with mp.Pool(workers) as pool:
    for subject_name, mean_facial_landmarks in pool.imap_unordered(subject_landmarks, jobs):
      per_subject[subject_name] = mean_facial_landmarks
      total_time = time.time() - start
      expected_time = total_time / len(per_subject) * len(list_subject_name)
      print('-' * 30)
      print(f'{subject_name} saved in path {os.path.join(out_folder, subject_name + ".pkl")}')
      print(f'Processed {len(per_subject)}/{len(list_subject_name)} subjects')
      print(f'Expected end: {int(expected_time/60/60)} h {int(expected_time/60)%60} m {int(expected_time)%60} s')
      print(f'Total time  : {int(total_time/60/60)} h {int(total_time/60)%60} m {int(total_time)%60} s')
      print('-' * 30)

  # Save mean landmarks of all subjects (summed in subject order, as before)
  all_mean_facial_landmarks = np.zeros((478, 3), dtype=np.float32)
  for subject_name in list_subject_name:
    all_mean_facial_landmarks += per_subject[subject_name]
  all_mean_facial_landmarks /= len(list_subject_name)
  dict_subject = {'mean_facial_landmarks': all_mean_facial_landmarks,
                  'count_subject': len(list_subject_name)}
  with open(os.path.join(out_folder, 'all_subjects_mean_landmarks.pkl'), 'wb') as f:
    pickle.dump(dict_subject, f)
  print(f'All subjects mean landmarks saved in {os.path.join(out_folder, "all_subjects_mean_landmarks.pkl")}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__.strip().split('\n\n')[0])
  parser.add_argument('--video_root', default='partA/video/video',
                      help='BioVid videos as <video_root>/<subject_name>/<sample_name>.mp4')
  parser.add_argument('--csv', default=os.path.join('partA', 'starting_point', 'samples.csv'),
                      help='BioVid label CSV (its subject_name column lists the subjects)')
  parser.add_argument('--out_folder', default='partA/video/mean_face_landmarks_per_subject',
                      help='Output folder of the per-subject and all-subject landmark pickles')
  parser.add_argument('--workers', type=int, default=1, help='Subjects processed in parallel')
  args = parser.parse_args()
  main(args.video_root, args.csv, args.out_folder, args.workers)
