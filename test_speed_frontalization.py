import custom.faceExtractor as extractor
import custom.tools as tools
import os 
import pickle
import time 

def load_reference_landmarks(path):
  ref_landmarks = pickle.load(open(path, 'rb'))
  ref_landmarks = ref_landmarks['mean_facial_landmarks']
  return ref_landmarks

face_extractor = extractor.FaceExtractor()
csv_path = os.path.join('partA','starting_point','samples.csv') 
# csv_array = tools.get_array_from_csv(csv_path=csv_path)
list_video_path = tools.get_list_video_path_from_csv(csv_path=csv_path)
ref_landmarks = load_reference_landmarks(os.path.join('partA', 'video', 'mean_face_landmarks_per_subject', 'all_subjects_mean_landmarks.pkl'))
frame_list = tools.get_list_frame_from_video_path(video_path=list_video_path[0])
list_frontalized_frames = []
start = time.time()
limit = 2
for frame in frame_list[:limit]:
  list_frontalized_frames.append(face_extractor.frontalize_img(frame=frame,
                                ref_landmarks=ref_landmarks,
                                time_logs=True) )
end = time.time()
print(f'Time to frontalized {limit} frames: {end-start:.2f} s')