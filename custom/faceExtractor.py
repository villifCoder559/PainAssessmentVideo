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
import copy
# from tools_face_elaboration import rigid_transform_3D, plot_landmarks_triangulation, get_frontalized_img

class FaceExtractor: 
  LEFT_EYE_INDEXES = [33, 133]  # Example: Left eye corners
  RIGHT_EYE_INDEXES = [362, 263]  # Example: Right eye corners
  NOSE_INDEX = 1  # Example: Nose tip
  FACE_OVAL = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
  FACE_TESSELATION = [(conn.start,conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
  RGB_IMAGE_BACKGROUND = (255, 0, 255)
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
      
    out = np.zeros_like(img) + 255
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
    # Create a copy of the image to draw on
    # if len(list_evidence_landmarks) == 0:
    #   list_evidence_landmarks = [self.NOSE_INDEX]
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
  
  # def generate_face_oval_video(self,path_video_input,path_video_output,align=False):
  #   routes_idx = self.FACE_OVAL
  #   new_video = []
  #   frame_list = self._get_list_frame(path_video_input,align=align)
  #   detection_result_list = self.extract_facial_landmarks(frame_list)
  #   start_time = time.time()
  #   for (img, _), detection_result in zip(frame_list, detection_result_list):
  #     landmarks = detection_result.face_landmarks[0]
  #     landmarks = self.center_wrt_nose(landmarks)
  #     routes = self._find_coords_point(routes_idx, landmarks, img)
  #     out_img,_ = self._extract_face_oval_from_img(img, routes)
  #     # print(f'out_img shape: {out_img.shape}')
  #     new_video.append(out_img)

  #   print(f'Time to generate face oval video: {time.time()-start_time} s')
  #   # print(f'dict_max_dim: {dict_max_dim}')
  #   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  #   fps = 25
  #   height, width,_ = new_video[0].shape
  #   print(f'height: {height}, width: {width}')
  #   if not os.path.exists(os.path.dirname(path_video_output)):
  #     os.makedirs(os.path.dirname(path_video_output))
  #   output_video = cv2.VideoWriter(path_video_output, fourcc, fps, (width, height))
  #   start_time = time.time()
  #   for frame in new_video:
  #     output_video.write(frame)
  #   output_video.release()
  #   print(f'Time to write video: {time.time()-start_time} s')
  #   print(f"Video saved at {path_video_output}")
  
  def frontalize_img(self,frame,ref_landmarks,frontalization_mode='SVD',align=True):                                                                             
    
    def compute_rigid_transform(A, B):
      rotation, translation = self.rigid_transform_3D(A=A.T, B=B.T)
      return rotation, translation
    
    def apply_rigid_transform(rotation, translation, landmarks):
      rot_trans_landmarks = rotation @ landmarks.T + translation
      return rot_trans_landmarks
    
    def estimate_affine_transform(landmarks, ref_landmarks):
      retval, affine_mat_3d, inliers = cv2.estimateAffine3D(landmarks, ref_landmarks)
      return affine_mat_3d
    
    def apply_affine_transform(landmarks, affine_mat_3d):
      cv_transfo_landmarks = cv2.transform(landmarks.reshape(1, -1, 3), affine_mat_3d).reshape(-1, 3)
      return cv_transfo_landmarks
    
    # face_extractor = extractor.FaceExtractor(visionRunningMode='image')
    if align:
      orig_frame = np.array(self.align_face(frame),dtype=np.uint8)
    else:
      orig_frame = np.copy(frame)
    # landmarks = self.extract_facial_landmarks([(frame, 0)])
    landmarks = self.extract_facial_landmarks([(orig_frame, 0)])
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0].face_landmarks[0]])
    # orig_frame,_ = self.extract_frame_oval_from_img(orig_frame,landmarks)
    
    # orig_frame,boolen_landmarks_mask = self.extract_frame_oval_from_img(orig_frame,landmarks)
    # landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0].face_landmarks[0]])
    if frontalization_mode == 'SVD':
      rotation, translation = compute_rigid_transform(landmarks, ref_landmarks)
      rot_trans_landmarks = apply_rigid_transform(rotation, translation, landmarks).T
      # landmarks_aligned_SVD = plot_landmarks_triangulation(image=np.zeros_like(orig_frame),
      #                                                                   landmarks=rot_trans_landmarks.T)
      frontalized_img_SVD = self._get_frontalized_img(landmarks_2d=landmarks,
                                            ref_landmarks_2d=rot_trans_landmarks,
                                            orig_frame=orig_frame)
      top_left_corner = (int(np.min(rot_trans_landmarks[:, 0]*frontalized_img_SVD.shape[1])), 
                         int(np.min(rot_trans_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
      bottom_right_corner = (int(np.max(rot_trans_landmarks[:, 0]*frontalized_img_SVD.shape[1])), 
                         int(np.max(rot_trans_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
      
      # cv2.fillConvexPoly(img, filler, 255)
      # print(f'top_left_corner: x {top_left_corner[0]}, y {top_left_corner[1]}')
      # print(f'bottom_right_corner: x {bottom_right_corner[0]}, y {bottom_right_corner[1]}')
      # cv2.circle(frontalized_img_SVD, top_left_corner, radius=4, color=(255, 255, 255), thickness=-1)
      # cv2.circle(frontalized_img_SVD, bottom_right_corner, radius=1, color=(255, 0, 255), thickness=-1)
      frontalized_img_SVD = self.post_process_frontalized_img(frontalized_img=frontalized_img_SVD,
                                                              top_left_corner=top_left_corner,
                                                              bottom_right_corner=bottom_right_corner,
                                                              landmarks=rot_trans_landmarks)

      return frontalized_img_SVD,rot_trans_landmarks
    
    elif frontalization_mode == 'CV2':
      affine_mat_3d = estimate_affine_transform(landmarks, ref_landmarks)
      cv_transfo_landmarks = apply_affine_transform(landmarks, affine_mat_3d)
      # landmarks_aligned_cv2 = self.plot_landmarks_triangulation(landmarks=cv_transfo_landmarks,
      #                                                                     image=np.zeros_like(orig_frame))
      frontalized_img_cv2 = self._get_frontalized_img(landmarks_2d=landmarks,
                                                ref_landmarks_2d=cv_transfo_landmarks,
                                                orig_frame=orig_frame)
      return frontalized_img_cv2,cv_transfo_landmarks
    else:
      raise ValueError(f'Invalid type: {frontalization_mode}. Can be "SVD" or "CV2"')

  # def post_process_frontalized_img(self,frontalized_img):
     
  def post_process_frontalized_img(self,frontalized_img,top_left_corner,bottom_right_corner,landmarks):
    # fill the image with face as much as possible
    # orig_image = copy.deepcopy(frontalized_img)
    print(f'landmarks shape {landmarks.shape}')
    landmarks = (landmarks * 256).astype(int)
    mask = np.zeros((frontalized_img.shape[0], frontalized_img.shape[1]))
    filler = cv2.convexHull(landmarks[:,:2])
    mask = cv2.fillConvexPoly(mask, filler, 1).astype(bool)
          
    # out = np.zeros_like(frontalized_img)
    # out[mask] = frontalized_img[mask]
    # fig, ax = plt.subplots(1,2, figsize=(12,12))
    # # print(f'img[mask] shape: {frontalized_img[dupl_mask].shape}')
    # ax[0].imshow(out)
    # ax[1].imshow(mask)

    center_pixel = (frontalized_img.shape[1]//2,frontalized_img.shape[0]//2)
    coords_face = np.argwhere(mask)
    count = 0
    print(f'len coords: {len(coords_face)}')
    for coord in coords_face:
      x = coord[1]
      y = coord[0]
      if frontalized_img[y,x][0] >= 240 or frontalized_img[y,x][1] >= 240 or frontalized_img[y,x][2] >= 240:
        count += 1
        # mirror the pixel
        mirror_x = center_pixel[0] - (x - center_pixel[0]) 
        frontalized_img[y,x] = frontalized_img[y,mirror_x]
    print(f'count changes: {count}')
    # bgr_frame = cv2.cvtColor(frontalized_img, cv2.COLOR_RGB2BGR)
    # gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    # _, bw_image = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
    # fig,ax = plt.subplots(1,2,figsize=(12,12))
    # ax[0].imshow(bw_image,cmap='gray')
    # ax[1].imshow(frontalized_img)
    # frontalized_img = cv2.inpaint(frontalized_img, bw_image, 3, cv2.INPAINT_TELEA)
    frontalized_img = frontalized_img[top_left_corner[1]:bottom_right_corner[1],top_left_corner[0]:bottom_right_corner[0]]
    frontalized_img=cv2.resize(frontalized_img,(256,256))
    
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
    for landmark in triangles:
      x = int(landmark[0])
      y = int(landmark[1])
      cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
    return image

  def apply_delaunay_triangulation(self,original_image, frontalized_landmarks, original_landmarks):
    # Step 1: Compute Delaunay triangulation for the original landmarks
    tri = Delaunay(original_landmarks)
    
    # Initialize the frontalized image
    frontalized_image = np.zeros_like(original_image)
    
    # Step 2: Iterate through each triangle
    for simplex in tri.simplices:
      # Get the vertex coordinates of the triangle in the original image
      original_triangle = np.float32(original_landmarks[simplex])*np.float32([original_image.shape[1], original_image.shape[0]])
      # Get the vertex coordinates of the triangle in the frontalized image
      frontalized_triangle = np.float32(frontalized_landmarks[simplex])*np.float32([frontalized_image.shape[1], frontalized_image.shape[0]])
      # Step 3: Compute the affine transformation matrix
      affine_transform,_ = cv2.estimateAffine2D(original_triangle, frontalized_triangle)
      # affine_transform = cv2.estimateAffine3D(original_triangle, frontalized_triangle)
      # original_image = plot_triangles_points(image=original_image, triangles=original_triangle)
      # for el in original_triangle:
      #   x = int(el[0])
      #   y = int(el[1])
      #   cv2.circle(original_image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
      # frontalized_image = plot_triangles_points(image=frontalized_image, triangles=frontalized_triangle)
      # for el in frontalized_triangle:
      #   x = int(el[0])
      #   y = int(el[1])
      #   cv2.circle(frontalized_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
      # Step 4: Apply the affine transformation to the triangle region
      # Create bounding boxes for the triangles
      triangle_mask = np.zeros_like(original_image)
      cv2.fillConvexPoly(triangle_mask, np.int32(frontalized_triangle), (255, 255, 255))
      # plt.imshow(triangle_mask)
      # Warp the region and copy it to the frontalized image
      warped_region = cv2.warpAffine(
        original_image, affine_transform, 
        (frontalized_image.shape[1], frontalized_image.shape[0]),
        # flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
      )
      frontalized_image = cv2.bitwise_and(frontalized_image, cv2.bitwise_not(triangle_mask))
      frontalized_image = cv2.bitwise_or(frontalized_image, cv2.bitwise_and(warped_region, triangle_mask))
      
    return frontalized_image,original_image

  def _get_frontalized_img(self,ref_landmarks_2d, landmarks_2d, orig_frame):
    if ref_landmarks_2d.shape[0] != landmarks_2d.shape[0]:
      raise ValueError(f'Number of landmarks do not match. ref_landmarks_2d: {ref_landmarks_2d.shape[0]}, landmarks_2d: {landmarks_2d.shape[0]}')
    if ref_landmarks_2d.shape[1] != 2:
      ref_landmarks_2d = ref_landmarks_2d[:, :2]
    if landmarks_2d.shape[1] != 2:
      landmarks_2d = landmarks_2d[:, :2]
    
    frontalized_img,_ = self.apply_delaunay_triangulation(original_image=orig_frame,
                                                  frontalized_landmarks=ref_landmarks_2d,
                                                  original_landmarks=landmarks_2d)
    return frontalized_img

  def plot_landmarks_triangulation(self,image,landmarks):
    img = np.copy(image)
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