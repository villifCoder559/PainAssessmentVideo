import os 
import numpy as np

log_path = os.path.join("partA", "video", "mean_face_landmarks_per_subject", "no_detection_log.txt") 
with open(log_path, "r") as f:
  no_detection = f.read().split("\n")
  no_detection = [x for x in no_detection if x != ""]
# split_log = np.array([x.split(",") for x in no_detection])
# print(split_log)
samples = np.array([x.split(",")[0] for x in no_detection])
frames = np.array([x.split(",")[1] for x in no_detection])
# print(f"Samples: {samples}")
# print(f"Frames: {frames}")
unique_samples = np.unique(samples)
list_dict = []
for sample in unique_samples:
  mask = samples == sample
  consecuteve_frames = np.diff(frames[mask].astype(int))
  if len(consecuteve_frames) == 0:
    continue
  max_consecutive_frames = np.max(consecuteve_frames)
  dict_sample = {
    'sample': sample,
    'frames': len(frames[mask]),
    'consecutive_frames': consecuteve_frames,
    'max_consecutive_frames': max_consecutive_frames
  }
  list_dict.append(dict_sample)
for el in list_dict:
  with open(os.path.join(os.path.split(log_path)[0], "no_detection_summary.txt"), "a") as f:
    f.write(f"\n{el['sample']}:\n")
  for k,v in el.items():
    with open(os.path.join(os.path.split(log_path)[0], "no_detection_summary.txt"), "a") as f:
      f.write(f" {k}: {v}\n")
      