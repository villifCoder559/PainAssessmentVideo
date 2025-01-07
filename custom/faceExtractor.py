import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceAligner,FaceLandmarksConnections
from pympler import asizeof
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FaceExtractor:
  LEFT_EYE_INDEXES = [33, 133]  # Example: Left eye corners
  RIGHT_EYE_INDEXES = [362, 263]  # Example: Right eye corners
  NOSE_INDEX = 1  # Example: Nose tip
  FACE_OVAL_ROUTE = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
  FACE_TESSELATION = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
  
  def __init__(self,
               face_extractor_options=None,
               base_options=python.BaseOptions(model_asset_path=os.path.join('landmark_model','face_landmarker.task'),
                                    delegate=mp.tasks.BaseOptions.Delegate.CPU),
               visionRunningMode=mp.tasks.vision.RunningMode):
    self.base_options = base_options
    self.visionRunningMode = visionRunningMode
    
    if face_extractor_options is None:
      self.options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1,
                                            running_mode=visionRunningMode.VIDEO,
                                            )
    else:
      self.options = face_extractor_options
      
    self.mp_face_aligner = mp.tasks.vision.FaceAligner.create_from_model_path(
                                                      model_path=base_options.model_asset_path)
    
  def align_face(self,mp_image):
    aligned_image = self.mp_face_aligner.align(mp_image)
    
    if aligned_image is None:
      print("No face detected.")
      return None
    
    aligned_image_np = aligned_image.numpy_view()
    # print(f'new img shape : {aligned_image_np.shape}')
    return aligned_image_np

  def _find_coords_point(self,routes_idx, landmarks, img):
    routes = []
    for source_idx, target_idx in routes_idx:
      source = landmarks[source_idx]
      target = landmarks[target_idx]
      relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
      relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

      #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
        
      routes.append(relative_source)
      routes.append(relative_target)
    return routes

  def _extract_face_oval_from_img(self,img,routes):
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
      
    out = np.zeros_like(img,dtype=np.uint8)
    out[mask] = img[mask]
    return out, None

  def _process_frame(self,detector,frame,timestamp):
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # print(int(timestamp))
    return detector.detect_for_video(mp_frame,int(timestamp))

  def _get_list_frame(self,path_video_input,align=False):
    cap = cv2.VideoCapture(path_video_input)
    frame_list = []
    if not cap.isOpened():
      raise IOError(f"Err: Unable to open video file: {path_video_input}")
    while cap.isOpened():
      ret, frame = cap.read()
      timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
      if not ret:
          break
      # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = np.array(frame,dtype=np.uint8)
      if align:
        img = self.align_face(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
      frame_list.append((img,timestamp_msec))
    cap.release()
    
    return frame_list
  
  def predict_landmarks(self,frame_list):
    """
    Predicts facial landmarks for a list of frames.

    Args:
      frame_list (list): A list of tuples, where each tuple contains a frame and its associated data.

    Returns:
      list: A list of detection results for each frame in the input list.
    """
    detector = vision.FaceLandmarker.create_from_options(self.options)
    detection_result_list = list(map(lambda x: self._process_frame(detector,x[0],x[1]) if x[0] is not None else None, frame_list))
    detector.close()
    return detection_result_list
  
  def center_wrt_nose(self,landmarks):
    nose = landmarks[self.NOSE_INDEX]
    list_centered_landmarks  = [mp.tasks.components.containers.NormalizedLandmark(x=-landmark.x + (nose.x+0.5),
                                                                                  y=-landmark.y + (nose.y+0.5),
                                                                                  z=-landmark.z + (nose.z+0.5)) for landmark in landmarks]

    return list_centered_landmarks
  
  def get_numpy_array(self,landmarks):
    return np.array([[lm.x,lm.y,lm.z] for lm in landmarks])
  
  def get_mean_facial_landmarks(self,list_video_path,align=True,numpy_view=True):
    all_landmarks = np.zeros(shape=(len(list_video_path),478,3),dtype=np.float32)
    count_frame = 0
    for count_video,video_path in enumerate(list_video_path):
      frame_list = self._get_list_frame(video_path,align=align)
      detection_result_list = self.predict_landmarks(frame_list)
      for frame_nr,detection_result in enumerate(detection_result_list):
        if detection_result is None or len(detection_result.face_landmarks) == 0:
          error_log_file = os.path.join('partA','video','mean_face_landmarks_per_subject','error.txt')
          if not os.path.exists(os.path.dirname(error_log_file)):
            os.makedirs(os.path.dirname(error_log_file))
          with open(error_log_file,'a') as f:
            f.write(f'No face detected: {video_path},frame_nr:{frame_nr}\n')
        else:
          # landmarks = self.get_numpy_array(detection_result.face_landmarks[0])
          landmarks = self.center_wrt_nose(detection_result.face_landmarks[0])
          # landmarks = self.convert_from_numpy_to_NormalizedLandmark(landmarks)
          count_frame += 1
          all_landmarks[count_video] += self.get_numpy_array(landmarks)
      if count_video+1 % 10 == 0:
        print(f'count_video: {count_video}')
    mean_face_landmarks = np.sum(all_landmarks,axis=0) / count_frame
    if numpy_view:
      return mean_face_landmarks,count_frame
    else:
      return [mp.tasks.components.containers.NormalizedLandmark(x=ln[0],y=ln[1],z=ln[2]) for ln in mean_face_landmarks],count_frame
      
  def convert_from_numpy_to_NormalizedLandmark(self,landmarks):
    return [mp.tasks.components.containers.NormalizedLandmark(x=ln[0],y=ln[1],z=ln[2]) for ln in landmarks]
    
  def generate_face_oval_video(self,path_video_input,path_video_output,align=False):
    routes_idx = self.FACE_OVAL_ROUTE
    new_video = []
    frame_list = self._get_list_frame(path_video_input,align=align)
    detection_result_list = self.predict_landmarks(frame_list)
    start_time = time.time()
    dict_max_dim = {'width': 0, 'height': 0}
    for (img, _), detection_result in zip(frame_list, detection_result_list):
      landmarks = detection_result.face_landmarks[0]
      landmarks = self.center_wrt_nose(landmarks)
      routes = self._find_coords_point(routes_idx, landmarks, img)
      out_img,_ = self._extract_face_oval_from_img(img, routes)
      # print(f'out_img shape: {out_img.shape}')
      new_video.append(out_img)

    print(f'Time to generate face oval video: {time.time()-start_time} s')
    # print(f'dict_max_dim: {dict_max_dim}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    height, width,_ = new_video[0].shape
    print(f'height: {height}, width: {width}')
    if not os.path.exists(os.path.dirname(path_video_output)):
      os.makedirs(os.path.dirname(path_video_output))
    output_video = cv2.VideoWriter(path_video_output, fourcc, fps, (width, height))
    start_time = time.time()
    for frame in new_video:
      output_video.write(frame)
    output_video.release()
    print(f'Time to write video: {time.time()-start_time} s')
    print(f"Video saved at {path_video_output}")
  
  def get_frames_annotated(self,path_video_input,align=False):
    """
    Extracts frames from a video, annotates them with facial landmarks, and returns the annotated frames.

    Args:
      path_video_input (str): The file path to the input video.
      align (bool, optional): If True, aligns the frames before processing. Defaults to False.

    Returns:
      list: A list of annotated frames, where each frame is an image with facial landmarks plotted.
    """
    frame_list = self._get_list_frame(path_video_input,align=align)
    detection_result_list = self.predict_landmarks(frame_list)
    new_video = []
    for (img, _), detection_result in zip(frame_list, detection_result_list):
      landmarks = detection_result.face_landmarks[0]
      landmarks = self.center_wrt_nose(landmarks)
      annotated_img = self.plot_landmarks(image=img,
                                          landmarks=landmarks,
                                          connections=self.FACE_TESSELATION)
      new_video.append(annotated_img)
    return new_video
  
  def warp_face(self,source_img, target_img, src_points, tgt_points):
    """Warp source image to match target landmarks."""
    # Compute affine transformation matrix
    M, _ = cv2.findHomography(src_points, tgt_points)
    
    # Warp source image
    warped_img = cv2.warpPerspective(source_img, M, (target_img.shape[1], target_img.shape[0]))
    
    return warped_img
  
  def plot_landmarks(self,image, landmarks, connections=None):
    """
    Plots facial landmarks on an image.
    Args:
        image (numpy.ndarray): The input image.
        landmarks (list): List of normalized landmarks with x, y, z coordinates.
        connections (list of tuple, optional): List of landmark connections to draw lines between points.
    """
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    height, width, _ = image.shape

    # Draw each landmark as a circle
    for idx,landmark in enumerate(landmarks):
      x = int(landmark.x * width)
      y = int(landmark.y * height)
      if idx == self.NOSE_INDEX:
        print(f'x: {x}, y: {y}')
        cv2.circle(annotated_image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
      else:
        cv2.circle(annotated_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    # Draw connections if provided
    if connections:
      for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
        cv2.line(annotated_image, start_point, end_point, color=(255, 0, 0), thickness=1)
    
    return annotated_image
    # Display the annotated image
    # cv2.imshow('Landmarks', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
