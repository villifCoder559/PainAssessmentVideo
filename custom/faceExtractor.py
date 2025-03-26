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
from scipy.signal import medfilt
# from tools_face_elaboration import rigid_transform_3D, plot_landmarks_triangulation, get_frontalized_img

class FaceExtractor:
  LEFT_CORNER_EYE_INDEXES = [33, 133]  # Example: Left eye corners
  RIGHT_CORNER_EYE_INDEXES = [362, 263]  # Example: Right eye corners
  NOSE_INDEX = 1  # Example: Nose tip
  FACE_OVAL = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
  FACE_TESSELATION = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
  
  # LEFT_EYE_INDEXES = [t[0] for t in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE]
  # RIGHT_EYE_INDEXES = [t[0] for t in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE]
  FACE_OVAL_INDEXES = [t[0] for t in FACE_OVAL]
  LEFT_EYEBROW_INDEXES = [t.start for t in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW]
  RIGHT_EYEBROW_INDEXES = [t.start for t in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW]
  CENTER_FACE_INDEXES = [10,151,9,8,168,66,197,195,5,4,1,19,194,164,0,11,12,15,16,17,18,200,199,175,152]
  STABLE_POINTS = [33,133,362,263,1]
  # TARGET = 199
  def __init__(self,
               min_face_detection_confidence=0.5,
               min_face_presence_confidence=0.5,
               min_tracking_confidence=0.5,
               num_faces=1,
               model_path = os.path.join('landmark_model','face_landmarker.task'),
               device = 'cpu',
               visionRunningMode='video',
               apply_mirroring_reconstruction=False):
    # running_mode video and num_faces = 1 mediapiep uses bult-in temporal smoothing
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
    self.mp_face_aligner = FaceAligner.create_from_options(options) # Create face aligner in image mode to call the align function
    self.config = {
      'min_face_detection_confidence': min_face_detection_confidence,
      'min_face_presence_confidence': min_face_presence_confidence,
      'min_tracking_confidence': min_tracking_confidence,
      'num_faces': num_faces,
      'model_path': model_path,
      'device': delegate,
      'visionRunningMode': running_mode,
      'apply_mirroring_reconstruction': apply_mirroring_reconstruction
    }
    self.apply_mirroring_reconstruction = apply_mirroring_reconstruction

    base_option_detector = python.BaseOptions(model_asset_path=os.path.join('landmark_model','mediapipe_detector.tflite'),
                                              # running_mode=running_mode,
                                              # min_detection_confidence=min_face_detection_confidence,
                                              # min_suppression_threshold = 0.3
                                              )
    
    options_detector = vision.FaceDetectorOptions(base_options=base_option_detector)
    self.face_detector = vision.FaceDetector.create_from_options(options_detector)
    self.landmarkers_detector = vision.FaceLandmarker.create_from_options(self.options)
    
  def process_detection_video(self, frame, timestamp, face_detector):
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    if self.config['visionRunningMode'] == mp.tasks.vision.RunningMode.IMAGE:
      return face_detector.detect(mp_frame)
    return face_detector.detect_for_video(mp_frame,int(timestamp))
    
  def crop_face_detection(self,image):
    # if not isinstance(image,mp.Image):
    #   image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # detection_result = self.face_detector.detect(image)
    # image = image.numpy_view()
    detection_result = self.extract_facial_landmarks(frame_list=[(image,0)])
    if detection_result[0] is None or len(detection_result[0].face_landmarks) == 0:
      print("No face detected.")
      return None
    landmarks = detection_result[0].face_landmarks
    landmarks = np.array([[lm.x,lm.y] for lm in landmarks[0]])
    top_left_corner = (int(np.min(landmarks[:, 0]*image.shape[1])),
                       int(np.min(landmarks[:, 1]*image.shape[0])))
    bottom_right_corner = (int(np.max(landmarks[:, 0]*image.shape[1])),
                           int(np.max(landmarks[:, 1]*image.shape[0])))
    # print(detection_result)
    # top_left_corner = (detection_result.detections[0].bounding_box.origin_x, detection_result.detections[0].bounding_box.origin_y)
    # bottom_right_corner = (top_left_corner[0] + detection_result.detections[0].bounding_box.width,
    #                       top_left_corner[1] + detection_result.detections[0].bounding_box.height)
    image = image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
    image = cv2.resize(image, (256, 256))
    image = np.array(image,dtype=np.uint8)
    # print(f'image shape: {image.shape}')
    return image
  def get_flatten_landmarks(self,ref_landmarks):
    shift = np.mean(ref_landmarks, axis=0)
    shift = ref_landmarks[1]
    centered_landmarks = [landmark - shift for landmark in ref_landmarks]
    centered_landmarks = np.array(centered_landmarks)

    # black_image = np.zeros((256,256,3),dtype=np.uint8)
    # tmp = ((centered_landmarks+0.5)*256).astype(np.int32)
    # orig_landmarks = face_extractor.plot_landmarks_triangulation(black_image,tmp)
    # fig,ax = plt.subplots(2,1,figsize=(10,10))
    # ax[0].set_title('Original landmarks')
    # ax[0].imshow(orig_landmarks)
    # print('Centered landmarks:',centered_landmarks.shape)
    # U,S,Vt = np.linalg.svd(centered_landmarks,full_matrices=False)
    # print('U:',U.shape)
    # print('S:',S.shape)
    # print('Vt:',Vt)
    landmarks_3d_norm = np.linalg.norm(centered_landmarks, axis=1,ord=2)
    landmarks_2d_norm = np.linalg.norm(centered_landmarks[:, :2], axis=1, ord=2)
    epsilon = 1e-6
    landmarks_2d_norm[landmarks_2d_norm == 0] = epsilon
    ratio = np.array(landmarks_3d_norm / landmarks_2d_norm, dtype=np.float32).reshape(-1, 1)
    centered_landmarks_2d = centered_landmarks[:, :2] * ratio
    return centered_landmarks_2d
    # frontalized_landmarks = self.plot_landmarks_triangulation(black_image,((centered_landmarks_2d+0.5)*256).astype(np.int32))
    # ax[1].set_title('Frontalized landmarks')
    # ax[1].imshow(frontalized_landmarks)
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

    out = np.zeros_like(img)
    out[mask] = img[mask]
    return out, mask

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
      frame = np.array(frame,dtype=np.uint8)
      if align:
        frame = self.align_face(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
      frame_list.append((frame,timestamp_msec))
    cap.release()

    return frame_list

  def extract_facial_landmarks(self,frame_list):
    """
    Predicts facial landmarks for a list of frames. It supposes that there is only one face in each frame.

    Args:
      frame_list (list): A list of tuples, where each tuple contains a frame and its associated data.

    Returns:
      list: A list of mediapipe NormalizedLandmarks object for each frame in the input list.
    """
    # if detector is None:
    #   detector = vision.FaceLandmarker.create_from_options(self.options)
    detection_result_list = []
    detector = vision.FaceLandmarker.create_from_options(self.options)
    for frame, timestamp in frame_list:
      if frame is not None:
        detection_result = self._process_frame(detector, frame, timestamp)
        np_landmarks = detection_result.face_landmarks[0] if len(detection_result.face_landmarks) > 0 else None
        if np_landmarks is not None:
          detection_result_list.append([[lm.x,lm.y,lm.z] for lm in np_landmarks])
        # detection_result_list.append(detection_result.face_landmarks[0] if len(detection_result.face_landmarks) > 0 else None)
      else:
        detection_result_list.append(None)
    return np.array(detection_result_list)

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
    if isinstance(landmarks,np.ndarray):
      landmarks = self.convert_from_numpy_to_NormalizedLandmark(landmarks)
    routes_idx = self.FACE_OVAL
    routes = self._find_coords_point(routes_idx, landmarks, img)
    out_img,mask = self._extract_face_oval_from_img(img, routes)
    return out_img,mask

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
    if isinstance(landmarks,np.ndarray):
      landmarks = self.convert_from_numpy_to_NormalizedLandmark(landmarks)
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

    # Draw connections if provided
    if connections:
      for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
        cv2.line(annotated_image, start_point, end_point, color=(0, 0, 255), thickness=1)

    return annotated_image,landmarks_coords
  
  def center_landmarks_wrt_nose(self,landmarks):
    nose = landmarks[self.NOSE_INDEX]
    
    if isinstance(landmarks,np.ndarray) and np.min(landmarks[:,:2])>=0 and np.max(landmarks[:,:2])<=1: # normalized landmarks
      # set nose landmarks in [0.5,0.5,0.5]
      centered_landmarks = landmarks - [nose-[0.5,0.5,0.5]]
    else:
      raise ValueError("Invalid landmarks format. Must be a list of NormalizedLandmark objects or a numpy array.")

    return centered_landmarks
  
  def center_frame_wrt_nose(self,frame,landmarks):
    nose = landmarks[self.NOSE_INDEX]
    centered_nose = (nose.x - 0.5, nose.y - 0.5)
    shift_x = int(centered_nose[0] * frame.shape[1])
    shift_y = int(centered_nose[1] * frame.shape[0])
    print(f'shift_x: {shift_x}, shift_y: {shift_y}')
    return shift_x,shift_y
  
  def frontalized_video(self,video_path,ref_landmarks,only_landmarks_crop=False,align_before_front=True,log_path=None,time_logs=False,extra_landmark_smoothing=None):

    def validate_frame_detection(list_to_validate):
      miss_detection = False
      for idx, el in enumerate(list_to_validate):
        list_no_detection_idx = []
        if el is None:
          list_no_detection_idx.append(idx)
          miss_detection = True
      return miss_detection, list_no_detection_idx

    # start = time.time()
    tuple_frames_timestamp = self._get_list_frame(video_path,align=align_before_front)
    # print("Time to get list frame: ",time.time()-start)

    miss_detection, list_no_detection_idx = validate_frame_detection(tuple_frames_timestamp)
    if miss_detection:
      raise DetectionError("No face detected in some frames during alignment", list_no_detection_idx)
    else:
      list_landmarks = self.extract_facial_landmarks(tuple_frames_timestamp)
      
      # list_landmarks = 
      miss_detection, list_no_detection_idx = validate_frame_detection(list_landmarks)
      if miss_detection:
        raise DetectionError("No face detected in some frames during landmarks detection", list_no_detection_idx)
      else:
        list_frontalized_img = []
        list_frontalized_landmarks = []
        list_frames = [frame for frame, _ in tuple_frames_timestamp]
        del tuple_frames_timestamp
        # print(f'Elapsed time to extract landmarks: {time.time()-start}')
        if extra_landmark_smoothing is not None and isinstance(extra_landmark_smoothing,LandmarkSmoother):
          print(f"Additional landmark smoothing: {extra_landmark_smoothing.method}")
          list_landmarks = extra_landmark_smoothing.smooth(list_landmarks)
        # start = time.time()
        if not only_landmarks_crop:
          for count, (frame, landmarks) in enumerate(zip(list_frames, list_landmarks)):
            # landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            # frame,mask = self.extract_frame_oval_from_img(frame,landmarks)
            rotation, translation = self.compute_rigid_transform(landmarks, ref_landmarks)
            frontalized_landmarks = self.apply_rigid_transform(rotation, translation, landmarks).T
            # rotation_stable, translation_stable = self.compute_rigid_transform(frontalized_landmarks[self.STABLE_POINTS], ref_landmarks[self.STABLE_POINTS])
            # print(f'Nose coords: {frontalized_landmarks[self.NOSE_INDEX]}')
            # frontalized_landmarks = self.apply_rigid_transform(rotation_stable, translation_stable, frontalized_landmarks).T
            # print(f'Nose coords: {frontalized_landmarks[self.NOSE_INDEX]}')
            
            frontalized_img_SVD = self._get_frontalized_img(landmarks_2d=landmarks,
                                                            frontalized_landmarks_2d=frontalized_landmarks,
                                                            orig_frame=frame,
                                                            log_path=log_path)

            top_left_corner = (int(np.min(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                              int(np.min(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
            bottom_right_corner = (int(np.max(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                              int(np.max(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))

            frontalized_img_SVD = self.post_process_frontalized_img(frontalized_img=frontalized_img_SVD,
                                                                      top_left_corner=top_left_corner,
                                                                      bottom_right_corner=bottom_right_corner,
                                                                      landmarks=frontalized_landmarks,
                                                                      )
            list_frontalized_img.append(frontalized_img_SVD)
            list_frontalized_landmarks.append(frontalized_landmarks)
        else:
          for count, (frame, landmarks) in enumerate(zip(list_frames, list_landmarks)):
            frame,mask = self.extract_frame_oval_from_img(frame,landmarks)
            top_left_corner = (int(np.min(landmarks[:, 0]*frame.shape[1])),
                              int(np.min(landmarks[:, 1]*frame.shape[0])))
            bottom_right_corner = (int(np.max(landmarks[:, 0]*frame.shape[1])),
                              int(np.max(landmarks[:, 1]*frame.shape[0])))
            frame = self.post_process_frontalized_img(frontalized_img=frame,
                                                      top_left_corner=top_left_corner,
                                                      bottom_right_corner=bottom_right_corner,
                                                      landmarks=landmarks)
            list_frontalized_img.append(frame)
            list_frontalized_landmarks.append(landmarks)
        return{
          'list_frontalized_frame': list_frontalized_img,
          'list_frontalized_landmarks': list_frontalized_landmarks,
        } 
  
  def frontalize_img(self,frame,ref_landmarks,align=True,time_logs=False,v2=False,stop_after=-1,log_path=None):
    start = time.time()
    if align:
      orig_frame = np.array(self.align_face(frame),dtype=np.uint8)
    else:
      orig_frame = np.copy(frame)

    landmarks = self.extract_facial_landmarks([(orig_frame, 0)])

    if landmarks[0] is not None and len(landmarks[0].face_landmarks) > 0:
      landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0].face_landmarks[0]])
      
      rotation, translation = self.compute_rigid_transform(landmarks, ref_landmarks)
      rot_trans_landmarks = self.apply_rigid_transform(rotation, translation, landmarks).T
      frontalized_img_SVD = self._get_frontalized_img(landmarks_2d=landmarks,
                                                      frontalized_landmarks_2d=rot_trans_landmarks,
                                                      orig_frame=orig_frame,
                                                      v2=v2,
                                                      stop_after=stop_after,
                                                      log_path=log_path)
        # print(f'TIme get frontal. img: {time.time()-time_transf:.4f}')
      top_left_corner = (int(np.min(rot_trans_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                        int(np.min(rot_trans_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
      bottom_right_corner = (int(np.max(rot_trans_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                        int(np.max(rot_trans_landmarks[:, 1]*frontalized_img_SVD.shape[0])))

      frontalized_img_SVD = self.post_process_frontalized_img(frontalized_img=frontalized_img_SVD,
                                                                top_left_corner=top_left_corner,
                                                                bottom_right_corner=bottom_right_corner,
                                                                landmarks=rot_trans_landmarks,
                                                                )
      return{
        'frontalized_img': frontalized_img_SVD,
        'frontalized_norm_landmarks': rot_trans_landmarks,
      } 
    else:
      print("No face detected")
      return None
  # def post_process_frontalized_img(self,frontalized_img):

  def compute_rigid_transform(self,A, B):
    rotation, translation = self.rigid_transform_3D(A=A.T, B=B.T)
    return rotation, translation

  def apply_rigid_transform(self,rotation, translation, landmarks):
    rot_trans_landmarks = rotation @ landmarks.T + translation
    return rot_trans_landmarks

  def estimate_affine_transform(self,landmarks, ref_landmarks):
    _, affine_mat_3d, _ = cv2.estimateAffine3D(landmarks, ref_landmarks)
    return affine_mat_3d

  def apply_affine_transform(self,landmarks, affine_mat_3d):
    cv_transfo_landmarks = cv2.transform(landmarks.reshape(1, -1, 3), affine_mat_3d).reshape(-1, 3)
    return cv_transfo_landmarks

  def post_process_frontalized_img(self,frontalized_img,top_left_corner,bottom_right_corner,landmarks):
    if self.apply_mirroring_reconstruction:
      landmarks = (landmarks * 256).astype(int) # from normalized to pixel coordinates
      mask = np.zeros((frontalized_img.shape[0], frontalized_img.shape[1]))
      filler = cv2.convexHull(landmarks[:,:2])
      mask = cv2.fillConvexPoly(mask, filler, 1).astype(bool)

      center_pixel = (frontalized_img.shape[1]//2,frontalized_img.shape[0]//2)
      coords_face = np.argwhere(mask)
      count = 0
      for coord in coords_face:
        x = coord[1]
        y = coord[0]
        if frontalized_img[y,x][0] >= 240 or frontalized_img[y,x][1] >= 240 or frontalized_img[y,x][2] >= 240:
          count += 1
          # mirror the pixel
          mirror_x = center_pixel[0] - (x - center_pixel[0])
          frontalized_img[y,x] = frontalized_img[y,mirror_x]
    frontalized_img = frontalized_img[top_left_corner[1]:bottom_right_corner[1],top_left_corner[0]:bottom_right_corner[0]]
    frontalized_img=cv2.resize(frontalized_img,(224,224))

    return frontalized_img

  def rigid_transform_3D(self,A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

  def plot_triangles_points(self,image, triangles):
    image = np.copy(image)
    for landmark in triangles:
      x = int(landmark[0])
      y = int(landmark[1])
      cv2.circle(image, (x, y), radius=4, color=(255, 0, 0), thickness=4)
    return image

  def plot_triangle(frame,points):
    for i in range(3):
      cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % 3]), (0, 255, 0), 1)
    return frame    

  def apply_delaunay_triangulation_v2(self, original_image, frontalized_landmarks, original_landmarks, stop_after=-1,log_path=None):
    # Pre-compute image dimensions scaling factors
    # h_orig, w_orig = original_image.shape[:2] - (1,1)
    h_orig, w_orig = original_image.shape[:2]
    h_front, w_front = original_image.shape[:2]
    scale_original = np.float32([w_orig, h_orig]) # Used to unscale the landmarks, so if landmarks is 1.0, it will be 255 (from 0 to img.shape-1) if it's 0 it will be 0
    scale_frontalized = np.float32([w_front, h_front]) # Used to unscale the landmarks, so if landmarks is 1.0, it will be 255 (from 0 to img.shape-1) if it's 0 it will be 0
    # Compute Delaunay triangulation for the frontalized landmarks
    tri = Delaunay(frontalized_landmarks)

    # Initialize the frontalized image
    frontalized_image = np.zeros_like(original_image)
    start = time.time()
    count = 0
    for idx,simplex in enumerate(tri.simplices):
      # Get vertex coordinates for the triangles
      original_triangle = (original_landmarks[simplex] * scale_original).astype(np.float32)
      frontalized_triangle = (frontalized_landmarks[simplex] * scale_frontalized).astype(np.float32)
      x_front, y_front, w_front, h_front = cv2.boundingRect(frontalized_triangle) # (x,y) top-left corner, (w,h) width and height
      x_orig_rect, y_orig_rect, w_orig_rect, h_orig_rect = cv2.boundingRect(original_triangle)
      delta = 16
      delta = delta if x_orig_rect-delta >= 0 and x_orig_rect + w_orig_rect + delta < w_orig  else min(x_orig_rect,w_orig - x_orig_rect - w_orig_rect +1) # +1 because right we have to consider that picture size is from 0 to w-1 
      delta = delta if y_orig_rect-delta >= 0 and y_orig_rect + h_orig_rect + delta < h_orig else min(y_orig_rect, h_orig - y_orig_rect - h_orig_rect +1) # +1 because bottom we have to consider that picture size is from 0 to h-1
      if delta < 0: # Mediapipe gives also >1 or < 0 when a part of the detected face is out of the image => delta can be negative
        delta = 0

      # bottom_y =
      normalized_frontalized_triangle = frontalized_triangle - [x_front, y_front] + [delta,delta]
      # print(f'normalized_frontalized_triangle: \n{normalized_frontalized_triangle}')
      normalized_original_triangle = original_triangle - [x_orig_rect, y_orig_rect] + [delta,delta]

      img_orig_cut = original_image[y_orig_rect-delta:y_orig_rect + h_orig_rect+delta, x_orig_rect-delta:x_orig_rect + w_orig_rect+delta]
      
      affine_transform_norm = cv2.getAffineTransform(np.array(normalized_original_triangle,dtype=np.float32),
                                                     np.array(normalized_frontalized_triangle,dtype=np.float32))
      
      wrp_region = cv2.warpAffine(img_orig_cut, affine_transform_norm, (w_front+delta, h_front+delta))
      wrp_region = wrp_region[delta:delta+h_front,delta:delta+w_front]
      
      rect_mask = np.zeros((h_front, w_front), dtype=np.uint8)
      # Create a mask for the current triangle
      triangle_in_rect = frontalized_triangle - [x_front, y_front]
      cv2.fillConvexPoly(rect_mask, np.int32(triangle_in_rect), (255))

      mask_expanded = rect_mask[:, :, np.newaxis]
      frontalized_image_y = [y_front, y_front + h_front]
      frontalized_image_x = [x_front, x_front + w_front]

      mask_expanded = mask_expanded[:frontalized_image_y[1] - frontalized_image_y[0],
                                    :frontalized_image_x[1] - frontalized_image_x[0]]
      wrp_region = wrp_region[:frontalized_image_y[1] - frontalized_image_y[0],
                                    :frontalized_image_x[1] - frontalized_image_x[0]]

      # adapt the mask to the frontalized image
      frontalized_image[frontalized_image_y[0]: frontalized_image_y[1], 
                        frontalized_image_x[0]: frontalized_image_x[1]] = (
          wrp_region * (mask_expanded / 255.0) +
          frontalized_image[frontalized_image_y[0]: frontalized_image_y[1],
                            frontalized_image_x[0]:frontalized_image_x[1]] * (1 - mask_expanded / 255.0)
      )
      
      # if count == tri.simplices.shape[0]-675:
      if log_path and count == tri.simplices.shape[0] - 1:
        img_front_landmarks,top_left_f,bottom_right_f = self.plot_landmarks_triangulation(image=np.zeros_like(frontalized_image),
                                                                                          landmarks=frontalized_landmarks,
                                                                                          tri_simplices=tri.simplices,
                                                                                          fill_triangle_idx=simplex,
                                                                                          padding=10)
        img_orig_landmarks,top_left_o,bottom_right_o = self.plot_landmarks_triangulation(image=np.zeros_like(original_image),
                                                                                        landmarks=original_landmarks,
                                                                                        tri_simplices=tri.simplices,
                                                                                        fill_triangle_idx=simplex,
                                                                                        padding=10)
        img_front_landmarks = img_front_landmarks[top_left_f[1]:bottom_right_f[1],top_left_f[0]:bottom_right_f[0]]
        img_orig_landmarks = img_orig_landmarks[top_left_o[1]:bottom_right_o[1],top_left_o[0]:bottom_right_o[0]]
        fig,ax = plt.subplots(2,2,figsize=(10,10))
        # set plot title
        plt.title('Delaunay triangulation v2')
        ax[0,0].set_title('frontalized image v2')
        fron_image_w_landmarks,_,_ = self.plot_landmarks_triangulation(image=frontalized_image,
                                                                    landmarks=frontalized_landmarks,
                                                                    tri_simplices=tri.simplices,
                                                                    fill_triangle_idx=simplex)
        # frontalized_image = cv2.resize(frontalized_image[top_left_f[1]:bottom_right_f[1],top_left_f[0]:bottom_right_f[0]],(190,155))
        # ax[0,0].imshow(frontalized_image)
        ax[0,0].imshow(fron_image_w_landmarks[top_left_f[1]:bottom_right_f[1],top_left_f[0]:bottom_right_f[0]])
        ax[0,1].set_title('original image')
        orig_image_w_landmarsks,_,_ = self.plot_landmarks_triangulation(image=original_image,
                                                                    landmarks=original_landmarks,
                                                                    tri_simplices=tri.simplices,
                                                                    fill_triangle_idx=simplex)
        ax[0,1].imshow(orig_image_w_landmarsks[top_left_o[1]:bottom_right_o[1],top_left_o[0]:bottom_right_o[0]])
        
        ax[1,0].set_title('frontalized landmarks')
        ax[1,0].imshow(img_front_landmarks)
        ax[1,1].set_title('original landmarks')
        ax[1,1].imshow(img_orig_landmarks)
        saving_path = log_path+f'_{count}_v2.png'
        fig.savefig(saving_path)
        plt.close()
        print(f'Saved image in {saving_path}')
      # print('count: ',count)
      # plt.savefig(os.path.join(log_path,f'{count}_v2.png'))
      count += 1
      # print(f'count: {count}')
      if count== stop_after:
        break
    return frontalized_image, original_image

  def apply_delaunay_triangulation(self,original_image, frontalized_landmarks, original_landmarks, log_path=None,stop_after=-1):
    # Step 1: Compute Delaunay triangulation for the original landmarks
    tri = Delaunay(frontalized_landmarks)
    # Initialize the frontalized image
    frontalized_image = np.zeros_like(original_image)

    # Step 2: Iterate through each triangle
    start = time.time()
    count = 0
    for simplex in tri.simplices:
      # print(f'simplex: {simplex}')  
      # Get the vertex coordinates of the triangle in the original image
      original_triangle = np.float32(original_landmarks[simplex])*np.float32([original_image.shape[1], original_image.shape[0]]) # shape(3,2)
      # Get the vertex coordinates of the triangle in the frontalized image
      frontalized_triangle = np.float32(frontalized_landmarks[simplex])*np.float32([frontalized_image.shape[1], frontalized_image.shape[0]]) # shape(3,2)
      # Compute the bounding rectangle for the frontalized triangle
      triangle_mask = np.zeros_like(original_image)
      tmp = time.time()
      affine_transform = cv2.getAffineTransform(original_triangle, frontalized_triangle)
      # print(f'affine_transform gt: {affine_transform}')
      cv2.fillConvexPoly(triangle_mask, np.int32(frontalized_triangle), (255, 255, 255))
      # plt.imshow(triangle_mask)
      # Warp the region and copy it to the frontalized image
      # original_image = self.plot_triangles_points(original_image, original_triangle)
      warped_region = cv2.warpAffine(
        original_image, affine_transform,
        (frontalized_image.shape[1], frontalized_image.shape[0]),
        # flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
      )

      frontalized_image = cv2.bitwise_and(frontalized_image, cv2.bitwise_not(triangle_mask))
      frontalized_image = cv2.bitwise_or(frontalized_image, cv2.bitwise_and(warped_region, triangle_mask))
      count += 1
      if count == stop_after:
        break
    # print(f'results: {results}')
    
    return frontalized_image,original_image

  def _get_frontalized_img(self,frontalized_landmarks_2d, landmarks_2d, orig_frame, v2=False,stop_after=-1,log_path=None):
    if frontalized_landmarks_2d.shape[1] < 2 or landmarks_2d.shape[1] < 2:
      raise ValueError("Invalid landmarks shape. Must be (n,2) or (n,3).")  
    if frontalized_landmarks_2d.shape[1] != 2:
      frontalized_landmarks_2d = frontalized_landmarks_2d[:, :2]
    if landmarks_2d.shape[1] != 2: 
      landmarks_2d = landmarks_2d[:, :2]
    
    frontalized_img,_ = self.apply_delaunay_triangulation_v2(original_image=orig_frame,
                                                  frontalized_landmarks=frontalized_landmarks_2d,
                                                  original_landmarks=landmarks_2d,
                                                  stop_after=stop_after,
                                                  log_path=log_path)
    return frontalized_img

  def plot_landmarks_triangulation(self,image,landmarks,tri_simplices=None,fill_triangle_idx=None,padding=10):
    img = np.copy(image)
    if landmarks.shape[1] > 2:
      landmarks = landmarks[:,:2]
    if np.max(landmarks) <= 1 and np.min(landmarks) >= 0:
      landmarks = landmarks * (img.shape[1],img.shape[0])
      landmarks = landmarks.astype(np.int32)
    if tri_simplices is None:
      tri = Delaunay(landmarks)
      tri_simplices = tri.simplices
    for idx,triangle in enumerate(tri_simplices):
      p1 = landmarks[triangle[0]]
      p2 = landmarks[triangle[1]]
      p3 = landmarks[triangle[2]]
      cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
      cv2.line(img, tuple(p2), tuple(p3), (0, 0, 255), 1)
      cv2.line(img, tuple(p3), tuple(p1), (0, 0, 255), 1)
      if fill_triangle_idx is not None:
        if np.intersect1d(triangle,fill_triangle_idx).shape[0] == 3:
          cv2.fillConvexPoly(img, np.int32([p1,p2,p3]), (255, 0, 0))
        
      # cv2.circle(img, tuple(p1), 2, (255, 0, 0), -1)
      # cv2.circle(img, tuple(p2), 2, (255, 0, 0), -1)
      # cv2.circle(img, tuple(p3), 2, (255, 0, 0), -1)
    top_left_corner = (np.min(landmarks[:, 0])-padding, np.min(landmarks[:, 1])-padding)
    bottom_right_corner = (np.max(landmarks[:, 0])+padding, np.max(landmarks[:, 1])+padding)
    # top_left_corner = (top_left_corner[0] - padding, top_left_corner[1] - padding)
    # bottom_right_corner = (bottom_right_corner[0] + padding, bottom_right_corner[1] + padding)
    return img,top_left_corner,bottom_right_corner
    # return img
    

class LandmarkSmoother:
  def __init__(self, method="kalman", window_size=5):
    """
    Initializes the faceExtractor with the specified method and window size.

    Args:
      method (str): The method to be used for smoothing. Can be "moving_average", "median_filter", or "kalman". 
      window_size (int): The size of the window for processing. Default is 5.
    """

    self.method = method
    self.window_size = window_size
    self.kalman_filters = []

  def initialize_kalman_filters(self, num_points):
    """Initialize a Kalman filter for each landmark point (x, y)."""
    self.kalman_filters = []
    for _ in range(num_points):
      kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, dx, dy), 2 measurements (x, y)
      kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
      kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
      kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
      self.kalman_filters.append(kf)

  def smooth(self, landmarks):
    """Apply smoothing to landmarks using the selected method."""
    landmarks = np.array(landmarks)  # Shape: (num_frames, num_landmarks, 2)
    num_frames, num_points, _ = landmarks.shape

    if self.method == "moving_average":
      return self.moving_average(landmarks)
    elif self.method == "median_filter":
      return self.median_filter(landmarks)
    elif self.method == "kalman":
      if not self.kalman_filters:
        self.initialize_kalman_filters(num_points)
      return self.kalman_filter_smoothing(landmarks)
    else:
      raise ValueError("Invalid smoothing method!")

  def moving_average(self, landmarks):
    """Apply a moving average filter to smooth landmarks."""
    smoothed = np.copy(landmarks)
    for i in range(len(landmarks)):
      smoothed[i] = np.mean(landmarks[max(0, i - self.window_size):i + 1], axis=0)
    return smoothed

  def median_filter(self, landmarks):
    """Apply a median filter to smooth landmarks."""
    return medfilt(landmarks, kernel_size=[self.window_size, 1, 1])

  def kalman_filter_smoothing(self, landmarks):
    """Smooth landmarks using Kalman filtering."""
    smoothed = np.zeros_like(landmarks)
    for point_idx in range(landmarks.shape[1]):  # Iterate over landmark points
      kf = self.kalman_filters[point_idx]
      for frame_idx in range(landmarks.shape[0]):
        measurement = np.array([[np.float32(landmarks[frame_idx, point_idx, 0])], 
                                [np.float32(landmarks[frame_idx, point_idx, 1])]])
        if frame_idx == 0:
          kf.statePre = np.array([[measurement[0, 0]], [measurement[1, 0]], [0], [0]], dtype=np.float32)
          kf.statePost = kf.statePre.copy()

        kf.correct(measurement)
        prediction = kf.predict()
        smoothed[frame_idx, point_idx] = prediction[:2].flatten()
    return smoothed

class DetectionError(Exception):
  def __init__(self, message, list_no_detection_idx):
    super().__init__(message)
    self.list_no_detection_idx = list_no_detection_idx
