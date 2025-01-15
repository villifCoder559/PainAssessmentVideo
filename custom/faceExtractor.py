import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceAligner,FaceLandmarksConnections,FaceAlignerOptions
from scipy.spatial import Delaunay
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FaceExtractor: 
  LEFT_EYE_INDEXES = [33, 133]  # Example: Left eye corners
  RIGHT_EYE_INDEXES = [362, 263]  # Example: Right eye corners
  NOSE_INDEX = 1  # Example: Nose tip
  FACE_OVAL = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
  FACE_TESSELATION = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
  # TARGET = 199
  def __init__(self,
               min_face_detection_confidence=0.5,
               min_face_presence_confidence=0.5,
               min_tracking_confidence=0.5,
               num_faces=1,
               model_path = os.path.join('landmark_model','face_landmarker.task'),
               device = 'cpu',
               visionRunningMode='video'):
    
    delegate = mp.tasks.BaseOptions.Delegate.CPU if device == 'cpu' else mp.tasks.BaseOptions.Delegate.GPU if device == 'gpu' else mp.tasks.BaseOptions.Delegate.CPU
    if device not in ['cpu','gpu']:
      print("Invalid device value, it can be either 'cpu' or 'gpu'. Set to 'cpu' by default.")
    
    running_mode = mp.tasks.vision.RunningMode.VIDEO if visionRunningMode == 'video' else mp.tasks.vision.RunningMode.IMAGE
    if visionRunningMode not in ['video', 'image']:
      print("Invalid running mode value, it can be either 'video' or 'image'. Set to 'image' by default.")
      
    base_options=python.BaseOptions(model_asset_path=model_path,
                                    delegate=delegate)
    self.base_options = base_options
    
    self.options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                output_face_blendshapes=False,
                                                output_facial_transformation_matrixes=False,
                                                num_faces=num_faces,
                                                min_face_detection_confidence=min_face_detection_confidence,
                                                min_face_presence_confidence=min_face_presence_confidence,
                                                min_tracking_confidence=min_tracking_confidence,
                                                running_mode=running_mode)
      
    options = FaceAlignerOptions(base_options=base_options)         # Enable refined landmarks
    self.mp_face_aligner = FaceAligner.create_from_options(options)
    self.config = {
      'min_face_detection_confidence': min_face_detection_confidence,
      'min_face_presence_confidence': min_face_presence_confidence,
      'min_tracking_confidence': min_tracking_confidence,
      'num_faces': num_faces,
      'model_path': model_path,
      'device': delegate,
      'visionRunningMode': running_mode
    }
    
  def align_face(self,image):
    """
    Aligns the face in the given MediaPipe image.
    This method takes a MediaPipe image, checks if it is a valid MediaPipe Image object,
    and aligns the face in the image using the MediaPipe face aligner. If no face is detected,
    it returns None.
    Args:
      mp_image (mp.Image or numpy.ndarray): The input image. It can be a MediaPipe Image object
      or a numpy array representing the image data.
    Returns:
      numpy.ndarray or None: The aligned face image as a numpy array if a face is detected,
      otherwise None.
    """
    #check if is mp image
    if not isinstance(image,mp.Image):
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    aligned_image = self.mp_face_aligner.align(image)
    if aligned_image is None:
      print("No face detected.")
      return None
    
    aligned_image_np = aligned_image.numpy_view()
    # print(f'new img shape : {aligned_image_np.shape}')
    return aligned_image_np

  def _find_coords_point(self,routes_idx, landmarks, img):
    """
    Calculate the coordinates of points based on given routes and landmarks.

    Args:
      routes_idx (list of tuples): A list of tuples where each tuple contains two indices representing the source and target points in the landmarks.
      landmarks (list): A list of landmark points, where each point has 'x' and 'y' attributes representing its coordinates.
      img (numpy.ndarray): The image on which the landmarks are to be mapped. The shape of the image is used to calculate the relative coordinates.

    Returns:
      list: A list of tuples representing the coordinates of the source and target points in the image.
    """
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
    """
    Extracts the face oval from the given image using the specified routes.
    Args:
      img (numpy.ndarray): The input image from which the face oval is to be extracted.
      routes (list): A list of points defining the convex polygon that represents the face oval.
    Returns:
      tuple: A tuple containing:
        - numpy.ndarray: The output image with the face oval extracted.
        - None: Placeholder for additional return value (currently not used).
    """
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
      
    out = np.zeros_like(img,dtype=np.uint8)
    out[mask] = img[mask]
    return out, None
  
  def _process_frame(self,detector,frame,timestamp):
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) 
    if self.config['visionRunningMode'] == mp.tasks.vision.RunningMode.IMAGE:
      return detector.detect(mp_frame)
    return detector.detect_for_video(mp_frame,int(timestamp))

  def _get_list_frame(self,path_video_input,align=False):
    """
    Extracts frames from a video file and returns them as a list along with their timestamps.
    Args:
      path_video_input (str): The path to the input video file.
      align (bool): If True, aligns the faces in the frames using the align_face method. Default is False.
    Returns:
      list: A list of tuples, where each tuple contains a frame (as a numpy array) and its corresponding timestamp in milliseconds.
    Raises:
      IOError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path_video_input)
    frame_list = []
    if not cap.isOpened():
      raise IOError(f"Err: Unable to open video file: {path_video_input}")
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
      img = np.array(frame,dtype=np.uint8)
      if align:
        img = self.align_face(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
      frame_list.append((img,timestamp_msec))
    cap.release()
    
    return frame_list
  
  def extract_facial_landmarks(self,frame_list):
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
    list_centered_landmarks  = [mp.tasks.components.containers.NormalizedLandmark(x=landmark.x - (nose.x-0.5),
                                                                                  y=landmark.y - (nose.y-0.5),
                                                                                  z=landmark.z - (nose.z-0.5)) for landmark in landmarks]

    return list_centered_landmarks
  
  def get_numpy_array(self,landmarks):
    return np.array([[lm.x,lm.y,lm.z] for lm in landmarks])
  
  def get_mean_facial_landmarks(self,list_video_path,align=True,numpy_view=True):
    all_landmarks = np.zeros(shape=(len(list_video_path),478,3),dtype=np.float32)
    count_frame = 0
    for count_video,video_path in enumerate(list_video_path):
      frame_list = self._get_list_frame(video_path,align=align)
      detection_result_list = self.extract_facial_landmarks(frame_list)
      for frame_nr,detection_result in enumerate(detection_result_list):
        if detection_result is None or len(detection_result.face_landmarks) == 0:
          error_log_file = os.path.join('partA','video','mean_face_landmarks_per_subject','no_detection_log.txt')
          if not os.path.exists(os.path.dirname(error_log_file)):
            os.makedirs(os.path.dirname(error_log_file))
          with open(error_log_file,'a') as f:
            f.write(f'{video_path},{frame_nr}\n')
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
    if landmarks.shape[1] == 3:
      return [mp.tasks.components.containers.NormalizedLandmark(x=ln[0],y=ln[1],z=ln[2]) for ln in landmarks]
    elif landmarks.shape[1] == 2:
      return [mp.tasks.components.containers.NormalizedLandmark(x=ln[0],y=ln[1]) for ln in landmarks]
    else:
      raise ValueError("Invalid landmarks shape. Must be (n,2) or (n,3).")
  
  def extract_frame_oval_from_img(self,img,landmarks):
    routes_idx = self.FACE_OVAL
    routes = self._find_coords_point(routes_idx, landmarks, img)
    out_img,_ = self._extract_face_oval_from_img(img, routes)
    return out_img  
  
  def generate_face_oval_video(self,path_video_input,path_video_output,align=False):
    routes_idx = self.FACE_OVAL
    new_video = []
    frame_list = self._get_list_frame(path_video_input,align=align)
    detection_result_list = self.extract_facial_landmarks(frame_list)
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
    detection_result_list = self.extract_facial_landmarks(frame_list)
    new_video = []
    for (img, _), detection_result in zip(frame_list, detection_result_list):
      landmarks = detection_result.face_landmarks[0]
      landmarks = self.center_wrt_nose(landmarks)
      annotated_img,_ = self.plot_landmarks(image=img,
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
  
  def plot_landmarks(self,image, landmarks, connections=None,list_evidence_landmarks=[]):
    """
    Plots facial landmarks on the given image.
    Args:
      image (numpy.ndarray): The input image on which landmarks are to be drawn.
      landmarks (list): A list of landmark objects, each containing x, y, and z coordinates.
      connections (list, optional): A list of tuples representing connections between landmarks. 
                      Each tuple contains two indices indicating the start and end points of the connection.
    Returns:
      tuple: A tuple containing:
        - annotated_image (numpy.ndarray): The image with landmarks and connections drawn.
        - landmarks_coords (list): A list of dictionaries containing the x, y, and z coordinates of each landmark.
    """
    # Create a copy of the image to draw on
    # if len(list_evidence_landmarks) == 0:
    #   list_evidence_landmarks = [self.NOSE_INDEX]
    annotated_image = image.copy()
    height, width, _ = image.shape
    landmarks_coords = []
    # Draw each landmark as a circle
    for idx,landmark in enumerate(landmarks):
      x = int(landmark.x * width)
      y = int(landmark.y * height)
      landmarks_coords.append({'x':landmark.x,'y':landmark.y,'z':landmark.z})
      if idx in list_evidence_landmarks:
        # print(f'Nose_coords x: {landmark.x}, y: {landmark.y}')
        cv2.circle(annotated_image, (x, y), radius=2, color=(0, 0, 255), thickness=5)
      # else:
      #   # if idx in self.LEFT_EYE_INDEXES:
      #   cv2.circle(annotated_image, (x, y), radius=2, color=(255, 0, 0), thickness=-1)
        # else:
        #   cv2.circle(annotated_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    # Draw connections if provided
    if connections:
      for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
        cv2.line(annotated_image, start_point, end_point, color=(255, 0, 0), thickness=1)
    
    return annotated_image,landmarks_coords
    # Display the annotated image
    # cv2.imshow('Landmarks', annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  
  def plot_landmarks_triangulation(self,image,landmarks):
    img = image.copy()
    if landmarks.shape[1] > 2:
      landmarks = landmarks[:,:2]
    if np.max(landmarks) <= 1 and np.min(landmarks) >= 0:
      landmarks = landmarks * (img.shape[1],img.shape[0])
      landmarks = landmarks.astype(np.int32)
    tri = Delaunay(landmarks)
    for triangle in tri.simplices:
      p1 = landmarks[triangle[0]]
      p2 = landmarks[triangle[1]]
      p3 = landmarks[triangle[2]]
      cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
      cv2.line(img, tuple(p2), tuple(p3), (0, 0, 255), 1)
      cv2.line(img, tuple(p3), tuple(p1), (0, 0, 255), 1)
      cv2.circle(img, tuple(p1), 2, (255, 0, 0), -1)
      cv2.circle(img, tuple(p2), 2, (255, 0, 0), -1)
      cv2.circle(img, tuple(p3), 2, (255, 0, 0), -1)
    return img
  
