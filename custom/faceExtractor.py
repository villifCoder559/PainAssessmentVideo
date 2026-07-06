"""
FaceExtractor Module

This module provides the FaceExtractor class for extracting and processing facial features from images and videos using MediaPipe and OpenCV.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceAligner, FaceLandmarksConnections, FaceAlignerOptions
from scipy.spatial import Delaunay
import os
import time
import numpy as np
import cv2
from scipy.signal import medfilt, savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import custom.tools as tools

# Constants for facial landmarks
LEFT_CORNER_EYE_INDEXES = [33, 133]  # Left eye corners
RIGHT_CORNER_EYE_INDEXES = [362, 263]  # Right eye corners
NOSE_INDEX = 1  # Nose tip
FACE_OVAL = [(conn.start, conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
FACE_TESSELATION = [(conn.start, conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]

# Fractional margin added on each side of the detected face box before cropping the
# stable ROI that gets fed to the FaceLandmarker (see the `preprocess` option). Mirrors
# PREPROCESS_MARGIN in crop_face_oval.py.
PREPROCESS_MARGIN = 0.7
def ensure_array(X):
  X = np.asarray(X, dtype=np.float64)
  if X.ndim != 2 or X.shape[1] not in (2,3):
    raise ValueError("Points must be (N,2) or (N,3)")
  return X

def smooth_boxes(boxes, frame_shape, smooth_window=15):
  """
  Temporally stabilize a per-frame sequence of boxes into constant-size, smoothly
  moving crops.

  Converts each box to (center, size), fills detection gaps (None entries) by linear
  interpolation from neighboring frames, low-pass filters the center trajectory with a
  zero-phase Savitzky-Golay filter, and applies one constant box size for the whole
  sequence (the max interpolated width/height). A per-frame independent box jitters
  frame-to-frame: as a preprocessing ROI it defeats the MediaPipe VIDEO-mode tracker
  and makes the normalized landmarks incomparable across frames; as an output crop it
  produces variable frame sizes that get stretch-resized by the video writer. Boxes
  are clamped in-bounds by shifting, never by shrinking, so every frame keeps
  identical dimensions.

  Args:
    boxes:         List of per-frame (x0, y0, x1, y1) pixel boxes, or None for frames
                   with no detection.
    frame_shape:   (H, W) of the frames the boxes live in, used for clamping.
    smooth_window: Savitzky-Golay window in frames for the center trajectory. Clamped
                   to the sequence length; sequences shorter than 5 frames are not
                   smoothed (gap-filling and the constant size still apply).

  Returns:
    List aligned with boxes: (x0, y0, x1, y1) int tuples, all with the same width and
    height. If every input box is None, returns a list of all None (caller falls back
    to the full frame).
  """
  n = len(boxes)
  if n == 0:
    return []
  det_idx = [i for i, b in enumerate(boxes) if b is not None]
  if not det_idx:
    return [None] * n
  h, w = frame_shape[:2]
  det = np.array([boxes[i] for i in det_idx], dtype=np.float64)
  t = np.arange(n)
  cx = np.interp(t, det_idx, (det[:, 0] + det[:, 2]) / 2.0)
  cy = np.interp(t, det_idx, (det[:, 1] + det[:, 3]) / 2.0)
  bw = np.interp(t, det_idx, det[:, 2] - det[:, 0])
  bh = np.interp(t, det_idx, det[:, 3] - det[:, 1])

  window = min(smooth_window, n)
  if window % 2 == 0:
    window -= 1
  if window >= 5:  # needs to exceed polyorder=2; below 5 the fit is (near) identity
    cx = savgol_filter(cx, window, polyorder=2)
    cy = savgol_filter(cy, window, polyorder=2)

  box_w = min(w, int(np.ceil(bw.max())))
  box_h = min(h, int(np.ceil(bh.max())))
  smoothed = []
  for i in range(n):
    x0 = int(round(cx[i] - box_w / 2.0))
    y0 = int(round(cy[i] - box_h / 2.0))
    x0 = min(max(x0, 0), w - box_w)
    y0 = min(max(y0, 0), h - box_h)
    smoothed.append((x0, y0, x0 + box_w, y0 + box_h))
  return smoothed

class FaceExtractor:
  """
  A class for extracting and processing facial features from images and videos.

  Attributes:
    config (dict): Configuration parameters for the face extractor.
    face_detector: MediaPipe face detector instance.
    mp_face_aligner: MediaPipe face aligner instance.
  """

  def __init__(self, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
               min_tracking_confidence=0.5, num_faces=1, model_path='landmark_model/face_landmarker.task',
               device='cpu', visionRunningMode='video', apply_mirroring_reconstruction=False,
               preprocess=True, preprocess_detector_model_path='landmark_model/blaze_face_short_range.tflite',
               preprocess_detection_confidence=0.5):
    """
    Initialize the FaceExtractor with the given parameters.

    Args:
      min_face_detection_confidence (float): Minimum confidence for face detection.
      min_face_presence_confidence (float): Minimum confidence for face presence.
      min_tracking_confidence (float): Minimum confidence for tracking.
      num_faces (int): Number of faces to detect.
      model_path (str): Path to the face landmark model.
      device (str): Device to use ('cpu' or 'gpu').
      visionRunningMode (str): Running mode ('video' or 'image').
      apply_mirroring_reconstruction (bool): Whether to apply mirroring reconstruction.
      preprocess (bool): If True, build a standalone face detector used to locate the
        face and crop a per-frame ROI before landmark extraction (helps with small/distant
        faces). See _compute_per_frame_face_rois and _get_list_frame.
      preprocess_detector_model_path (str): Legacy/unused. Retained for backward
        compatibility; the preprocessing crop now uses the full-range Solutions
        BlazeFace detector (model_selection=1), which needs no model file.
      preprocess_detection_confidence (float): Minimum detection confidence for the
        preprocessing detector. Kept low to maximize small-face recall.
    """
    delegate = mp.tasks.BaseOptions.Delegate.CPU if device == 'cpu' else mp.tasks.BaseOptions.Delegate.GPU
    running_mode = mp.tasks.vision.RunningMode.VIDEO if visionRunningMode == 'video' else mp.tasks.vision.RunningMode.IMAGE

    base_options = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
    self.options = vision.FaceLandmarkerOptions(
      base_options=base_options,
      output_face_blendshapes=False,
      output_facial_transformation_matrixes=False,
      num_faces=num_faces,
      min_face_detection_confidence=min_face_detection_confidence,
      min_face_presence_confidence=min_face_presence_confidence,
      min_tracking_confidence=min_tracking_confidence,
      running_mode=running_mode
    )

    aligner_options = FaceAlignerOptions(base_options=base_options)
    self.mp_face_aligner = FaceAligner.create_from_options(aligner_options)
    self.face_detector = vision.FaceLandmarker.create_from_options(self.options)

    # Standalone face detector for the preprocessing ROI crop. Uses the full-range
    # Solutions BlazeFace detector (model_selection=1) to locate small/distant faces,
    # mirroring crop_face_oval.py. Only created when preprocess=True.
    self.preprocess = preprocess
    self.preprocess_detection_confidence = preprocess_detection_confidence
    self.preprocess_detector = None
    if preprocess:
      self.preprocess_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=preprocess_detection_confidence)

    self.FACE_OVAL = FACE_OVAL
    self.config = {
      'min_face_detection_confidence': min_face_detection_confidence,
      'min_face_presence_confidence': min_face_presence_confidence,
      'min_tracking_confidence': min_tracking_confidence,
      'num_faces': num_faces,
      'model_path': model_path,
      'device': device,
      'visionRunningMode': visionRunningMode,
      'apply_mirroring_reconstruction': apply_mirroring_reconstruction,
      'preprocess': preprocess,
      'preprocess_detector_model_path': preprocess_detector_model_path,
      'preprocess_detection_confidence': preprocess_detection_confidence
    }

  def align_face(self, image):
    """
    Align the face in the given image.

    Args:
      image (np.ndarray): Input image.

    Returns:
      np.ndarray: Aligned face image.
    """
    if not isinstance(image, mp.Image):
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    aligned_image = self.mp_face_aligner.align(image)
    if aligned_image is None:
      raise ValueError("No face detected in the image.")

    return aligned_image.numpy_view()

  def reset_face_detector(self):
    self.face_detector.close()
    self.face_detector = vision.FaceLandmarker.create_from_options(self.options)
    if self.preprocess_detector is not None:
      self.preprocess_detector.close()
      self.preprocess_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=self.preprocess_detection_confidence)

  def _compute_per_frame_face_rois(self, frame_list, margin=PREPROCESS_MARGIN):
    """
    Detect the face in every frame and return one tight, padded ROI per frame.

    Runs the full-range Solutions face detector on each frame, takes the top
    (highest-score) detection, and returns its bounding box expanded by `margin`
    on each side and clamped to the frame bounds. Cropping every frame to its own
    ROI enlarges the face for the downstream FaceLandmarker, which is what it needs
    on videos where the face is small relative to the frame. A single ROI unioned
    across all frames balloons to nearly the full frame when the face moves and
    defeats the zoom, so we crop per frame instead (mirrors crop_face_oval.py).

    Args:
      frame_list: List of RGB frames. Each has shape (H, W, 3), dtype uint8. All frames
                  are assumed to share the same (H, W).
      margin:     Fractional margin added on each side of each box (0.0 = tight).

    Returns:
      List aligned with frame_list; each element is a (x0, y0, x1, y1) tuple of pixel
      corners, or None for a frame with no detection (caller then falls back to the
      full frame for that frame).
    """
    if not frame_list or self.preprocess_detector is None:
      return [None] * len(frame_list)
    h, w = frame_list[0].shape[:2]
    rois = []
    for frame in frame_list:
      # frame is already RGB (see _get_list_frame), so it is passed straight to the
      # Solutions detector without a colour conversion.
      result = self.preprocess_detector.process(frame)
      if not result.detections:
        rois.append(None)
        continue
      box = result.detections[0].location_data.relative_bounding_box
      x0 = max(0, int((box.xmin - box.width * margin) * w))
      y0 = max(0, int((box.ymin - box.height * margin) * h))
      x1 = min(w, int((box.xmin + box.width * (1 + margin)) * w))
      y1 = min(h, int((box.ymin + box.height * (1 + margin)) * h))
      rois.append((x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None)
    return rois

  def extract_facial_landmarks(self, frame_list):
    """
    Extract facial landmarks from a list of frames.

    Args:
      frame_list (list): List of frames.

    Returns:
      list: List of facial landmarks.
    """
    detection_result_list = []
    for frame, timestamp in tqdm.tqdm(frame_list, desc="Extracting facial landmarks...", total=len(frame_list)):
      if frame is not None:
        detection_result = self.face_detector.detect_for_video(
          mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), int(timestamp))
        if detection_result.face_landmarks:
          detection_result_list.append([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0]])
        else:
          detection_result_list.append(None)
      else:
        detection_result_list.append(None)
    return detection_result_list

  def interpolate_frames_linear(frames, num_interpolations=1):
    interpolated_frames = []
    for i in range(len(frames) - 1):
      frame_a = frames[i]
      frame_b = frames[i + 1]
      interpolated_frames.append(frame_a)
      for j in range(1, num_interpolations + 1):
        alpha = j / (num_interpolations + 1)
        beta = 1.0 - alpha
        interpolated = cv2.addWeighted(frame_a, beta, frame_b, alpha, 0)
        interpolated_frames.append(interpolated)
    interpolated_frames.append(frames[-1])
    return interpolated_frames

  def interpolate_frame(A, B, t=0.5):
    # 1. Optical flow (e.g. Farneback)
    flow_A2B = cv2.calcOpticalFlowFarneback(cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(B,cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_B2A = cv2.calcOpticalFlowFarneback(cv2.cvtColor(B,cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = A.shape[:2]
    # 2. Build remap grids
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mapAx = (xs + flow_A2B[...,0] * t).astype(np.float32)
    mapAy = (ys + flow_A2B[...,1] * t).astype(np.float32)
    mapBx = (xs + flow_B2A[...,0] * (1-t)).astype(np.float32)
    mapBy = (ys + flow_B2A[...,1] * (1-t)).astype(np.float32)
    # 3. Warp with bilinear interpolation
    warpA = cv2.remap(A, mapAx, mapAy, cv2.INTER_LINEAR)
    warpB = cv2.remap(B, mapBx, mapBy, cv2.INTER_LINEAR)
    # 4. Linear blend
    return cv2.addWeighted(warpA, 1-t, warpB, t, 0)
    

  def crop_face_detection(self,image):
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

    landmarks_3d_norm = np.linalg.norm(centered_landmarks, axis=1,ord=2)
    landmarks_2d_norm = np.linalg.norm(centered_landmarks[:, :2], axis=1, ord=2)
    epsilon = 1e-6
    landmarks_2d_norm[landmarks_2d_norm == 0] = epsilon
    ratio = np.array(landmarks_3d_norm / landmarks_2d_norm, dtype=np.float32).reshape(-1, 1)
    centered_landmarks_2d = centered_landmarks[:, :2] * ratio
    return centered_landmarks_2d

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

    out = np.zeros_like(img)
    out[mask] = img[mask]
    return out, mask
  
  def _process_frame(self,detector,frame,timestamp):
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    if self.config['visionRunningMode'] == mp.tasks.vision.RunningMode.IMAGE:
      return detector.detect(mp_frame)
    return detector.detect_for_video(mp_frame,int(timestamp))

  def apply_video_interpolation(self,frame_list,chunk_size,fps=None,mod='mirror_start_video',landmarks_list=None):
    """
    Pad a video so its frame count reaches the next multiple of chunk_size.

    Args:
      frame_list:     List of frames, each an array of shape (H, W, 3).
      chunk_size:     Chunk size; frames are padded up to the next multiple of this value.
                      Videos whose length is already a multiple are returned unchanged.
      fps:            Frames per second, used to rebuild evenly spaced timestamps (ms).
      mod:            Padding modality: 'mirror_start_video' prepends a symmetric (ping-pong)
                      reflection of the first frames; 'spread_linearly' inserts averaged frames
                      spread across the video (no landmark support).
      landmarks_list: Optional per-frame landmarks list, padded with the same indices as the
                      frames ('mirror_start_video' only).

    Returns:
      (new_frame_list, new_timestamp_list) or, when landmarks_list is given,
      (new_frame_list, new_timestamp_list, new_landmarks_list).
    """
    new_frame_list = list(frame_list)
    new_video_frames = -(-len(frame_list) // chunk_size) * chunk_size
    frames_to_add = new_video_frames - len(frame_list)
    new_timestamp_list = None
    if fps is not None:
      step_timestamp_ms = int(1000 // fps)
      new_timestamp_list = list(range(0,new_video_frames*step_timestamp_ms,step_timestamp_ms))

    if mod == 'spread_linearly':
      if landmarks_list is not None:
        raise NotImplementedError("Landmarks interpolation not implemented for 'spread_linearly' mode.")
      if frames_to_add > 0:
        insert_after = np.round(np.linspace(1, max(len(frame_list)-2,0), num=frames_to_add)).astype(int)
        new_frame_list = []
        for count, frame in enumerate(frame_list):
          new_frame_list.append(frame)
          next_frame = frame_list[min(count+1, len(frame_list)-1)]
          for _ in range(int(np.sum(insert_after == count))):
            tmp = np.stack([frame,next_frame],axis=0).astype(np.float64)
            new_frame_list.append(np.mean(tmp,axis=0).astype(np.uint8))
    elif mod == 'mirror_start_video':
      # Symmetric (ping-pong) reflection of the start, as in np.pad(mode='symmetric'):
      # [f1, f0] + [f0, f1, ...]. Cycles through the video again when frames_to_add
      # exceeds its length (very short videos).
      n = len(frame_list)
      period = 2 * n
      prefix_idx = []
      for i in range(frames_to_add):
        j = (-(i + 1)) % period
        prefix_idx.append(j if j < n else period - 1 - j)
      prefix_idx.reverse()
      new_frame_list = [frame_list[j] for j in prefix_idx] + list(frame_list)
      if landmarks_list is not None:
        landmarks_list = [landmarks_list[j] for j in prefix_idx] + list(landmarks_list)
    else:
      raise ValueError("Invalid interpolation mode. Choose 'spread_linearly' or 'mirror_start_video'.")
    
    # return tuple for face frontalization function
    if landmarks_list is not None:
      return new_frame_list,new_timestamp_list,landmarks_list
    else:
      return new_frame_list,new_timestamp_list

  def _get_list_frame(self,path_video_input,return_tuple=True,align=False,preprocess=False,stabilize=True):
    """
    Read every frame of a video, optionally crop to a stable face ROI and align.

    When `preprocess` is True the raw frames are read first, a tight face ROI is computed
    per frame (see _compute_per_frame_face_rois) and every frame is cropped to its own ROI
    before the (optional) alignment step. This helps the FaceLandmarker on videos where the
    face is small relative to the frame. With `preprocess` False the behavior matches the
    original per-frame read (alignment simply runs in its own pass).

    Args:
      path_video_input: Path to the source video.
      return_tuple:     If True, return list of (frame, timestamp_ms) tuples. If False,
                        return (frame_list, timestamp_list, FPS).
      align:            If True, align each frame with the MediaPipe FaceAligner.
      preprocess:       If True, crop every frame to a stable face ROI before alignment.
      stabilize:        If True (default), temporally smooth the per-frame ROIs into
                        constant-size, smoothly tracking crops (see smooth_boxes). Raw
                        per-frame detector boxes jitter, which defeats the VIDEO-mode
                        landmark tracking. False restores the old per-frame boxes.

    Returns:
      If return_tuple: List[Tuple[np.ndarray, float]] of (RGB frame, timestamp in ms).
      Else: Tuple[List[np.ndarray], List[float], float] of (frames, timestamps_ms, FPS).
    """
    cap = cv2.VideoCapture(path_video_input)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []
    if not cap.isOpened():
      raise IOError(f"Err: Unable to open video file: {path_video_input}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=total_frames if total_frames > 0 else None, desc="Reading video frames...")
    # First pass: read all raw RGB frames.
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = np.array(frame,dtype=np.uint8)
      frame_list.append(frame)
      pbar.update(1)

    pbar.close()
    cap.release()

    # Optional preprocessing: crop every frame to its own tight face ROI so the face is
    # enlarged for the downstream landmarker (small-face videos). A per-frame crop is used
    # (not one shared ROI) because a union across a moving face balloons to the full frame
    # and defeats the zoom.
    if preprocess:
      rois = self._compute_per_frame_face_rois(frame_list)
      if all(roi is None for roi in rois):
        print(f"No face detected for preprocessing ROI in {path_video_input}; using full frames.")
      elif stabilize:
        # Replace the raw jittery per-frame boxes with constant-size, smoothly moving
        # crops; detection gaps are interpolated instead of falling back to the full
        # frame (which caused a one-frame scale jump).
        rois = smooth_boxes(rois, frame_list[0].shape[:2])
      cropped_list = []
      for frame, roi in zip(frame_list, rois):
        if roi is None:
          cropped_list.append(frame)  # no detection: fall back to the full frame
        else:
          x0, y0, x1, y1 = roi
          # np.ascontiguousarray: slicing yields a non-contiguous view, but mp.Image
          # requires a C-contiguous uint8 buffer.
          cropped_list.append(np.ascontiguousarray(frame[y0:y1, x0:x1]))
      frame_list = cropped_list

    # Optional alignment pass (runs on the possibly-cropped frames).
    if align:
      frame_list = [self.align_face(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
                    for frame in frame_list]

    timestamp_list = [(count / FPS) * 1000 for count in range(len(frame_list))]
    if return_tuple:
      return list(zip(frame_list,timestamp_list))
    else:
      return frame_list,timestamp_list,FPS


  def center_wrt_nose(self,landmarks):
    nose = landmarks[NOSE_INDEX]
    
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
    # check if routes is empty
    if len(routes) == 0:
      print("No routes found.")
      return None,None
    out_img,mask = self._extract_face_oval_from_img(img, routes)
    return out_img,mask

  def umeyama(self,P, Q, with_scale=True):
    """
    Umeyama: returns scale s (1 if with_scale=False), rotation R, and translation t.
    P, Q: (N, d) arrays
    """
    P = ensure_array(P)
    Q = ensure_array(Q)
    if P.shape != Q.shape:
      raise ValueError("P and Q must have the same shape")
    N, d = P.shape
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    X = P - mu_P
    Y = Q - mu_Q
    sigma2 = (X**2).sum() / N
    cov = (X.T @ Y) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
      S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
      scale = np.trace(np.diag(D) @ S) / sigma2
    else:
      scale = 1.0
    t = mu_Q - scale * R @ mu_P
    return scale, R, t

  def apply_transform(self,P, R, t, scale=1.0):
      P = ensure_array(P)
      return (scale * (P @ R.T)) + t  # note: P @ R^T => (R @ P^T)^T

  def generate_face_oval_video(self, path_video_input, path_video_output=None, generate_video=False,interpolation_mod_chunk=None, align=False, template_landmarks=None, preprocess=True):
    """
    Generate a face-oval video. If template_landmarks is provided, it should be a normalized
    (x,y) array of shape (N,2) with the same landmark ordering/indices used by self.FACE_OVAL.
    Alignment is applied per-frame using Umeyama similarity transform (scale + rotation + translation).

    When `preprocess` is True (default), every frame is cropped to a stable face ROI
    before landmark extraction (see _get_list_frame), which helps on small-face videos.
    """
    routes_idx = self.FACE_OVAL
    new_video = []
    frame_list,timestamp_list,FPS = self._get_list_frame(path_video_input, align=align,return_tuple=False,preprocess=preprocess)
    frame_list_timestamp = list(zip(frame_list,timestamp_list))
    detection_result_list = self.extract_facial_landmarks(frame_list_timestamp)
    
    if interpolation_mod_chunk is not None and len(interpolation_mod_chunk) == 2:
      frame_list_filtered = []
      timestamp_list_filtered = []
      detection_result_filtered = []
      for (frame,timestamp), detection in zip(frame_list_timestamp, detection_result_list):
        if detection is not None:
          frame_list_filtered.append(frame)
          timestamp_list_filtered.append(timestamp)
          detection_result_filtered.append(detection)
      frame_list = frame_list_filtered
      timestamp_list = timestamp_list_filtered
      detection_result_list = detection_result_filtered
      
      frame_list, timestamp_list, detection_result_list = self.apply_video_interpolation(frame_list=frame_list,
                                                                    mod=interpolation_mod_chunk[0],
                                                                    chunk_size=interpolation_mod_chunk[1],
                                                                    fps=FPS,
                                                                    landmarks_list=detection_result_list
                                                                    )
      frame_list_timestamp = list(zip(frame_list,timestamp_list))
      
      # self.reset_face_detector()
      # detection_result_list = self.extract_facial_landmarks(frame_list_timestamp)
    
    # start_time = time.time()
    # ---- main loop ----
    for (img, timestamp), frame_landmarks in tqdm.tqdm(
      zip(frame_list_timestamp, detection_result_list),
      total=len(frame_list_timestamp),
      desc="Generating face oval video..."):
      if frame_landmarks is None:
        print(f"No landmarks for frame, skipping. {path_video_input} at timestamp {timestamp}ms")
        continue
      # convert to normalized landmarks object (your function)
      landmarks = self.convert_from_numpy_to_NormalizedLandmark(np.array(frame_landmarks))
      routes = self._find_coords_point(routes_idx, landmarks, img)
      frame, _ = self._extract_face_oval_from_img(img, routes)

      # convert to Nx2 or Nx3 numpy array in normalized coords
      landmarks = self.get_numpy_array(landmarks)  # expected shape (N,2) or (N,3)
      h, w = frame.shape[0], frame.shape[1]

      # If template_landmarks argument is provided and valid, align landmarks using Umeyama
      if template_landmarks is not None:
        try:
          template = np.asarray(template_landmarks, dtype=np.float64)
          # require template to be normalized x,y in [0,1] with at least as many points as max index
          # max_idx = int(np.max(routes_idx))
          if template.ndim == 2 and template.shape[1] >= 2:
            # select corresponding points via routes_idx
            src_norm = landmarks[:, :2]   # FIX: Use routes_idx
            tgt_norm = template[:, :2]    # FIX: Use routes_idx

            # convert to pixel coords for better numerical stability
            src_px = np.empty_like(src_norm)
            tgt_px = np.empty_like(tgt_norm)
            src_px[:, 0] = src_norm[:, 0] * w
            src_px[:, 1] = src_norm[:, 1] * h
            tgt_px[:, 0] = tgt_norm[:, 0] * w
            tgt_px[:, 1] = tgt_norm[:, 1] * h

            # estimate similarity transform
            s, R, tvec = self.umeyama(src_px, tgt_px, with_scale=True)

            # FIX: Apply transform to the IMAGE, not just landmarks
            M = np.eye(3)
            M[:2, :2] = s * R
            M[:2, 2] = tvec
            frame = cv2.warpAffine(frame, M[:2, :], (w, h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

            # Also transform ALL landmarks for consistent crop calculation
            all_xy = landmarks[:, :2].copy()
            all_px = np.empty_like(all_xy)
            all_px[:, 0] = all_xy[:, 0] * w
            all_px[:, 1] = all_xy[:, 1] * h
            all_aligned_px = self.apply_transform(all_px, R, tvec, scale=s)
            
            aligned_norm = np.empty_like(landmarks[:, :2])
            aligned_norm[:, 0] = all_aligned_px[:, 0] / float(w)
            aligned_norm[:, 1] = all_aligned_px[:, 1] / float(h)

            if landmarks.shape[1] == 3:
              # keep original z channel
              landmarks = np.hstack([aligned_norm, landmarks[:, 2:3]])
            else:
              landmarks = aligned_norm
          else:
            # Template doesn't match required indices — skip alignment
            pass
        except Exception as e:
          # if anything fails, continue with original landmarks but log
          print(f"Umeyama alignment failed for frame at {timestamp}ms: {e}")

      # compute crop area (top-left and bottom-right) in pixels using normalized landmarks
      top_left_corner = (int(np.min(landmarks[:, 0] * frame.shape[1])),
                        int(np.min(landmarks[:, 1] * frame.shape[0])))
      bottom_right_corner = (int(np.max(landmarks[:, 0] * frame.shape[1])),
                            int(np.max(landmarks[:, 1] * frame.shape[0])))

      # clamp to image bounds
      top_left_corner = (max(0, top_left_corner[0]), max(0, top_left_corner[1]))
      bottom_right_corner = (min(frame.shape[1], bottom_right_corner[0]),
                            min(frame.shape[0], bottom_right_corner[1]))

      # if frame.size == 0:
      #   print(f'Frame is empty')
      frame = self.post_process_frontalized_img(frontalized_img=frame,
                                                top_left_corner=top_left_corner,
                                                bottom_right_corner=bottom_right_corner,
                                                landmarks=landmarks)
      new_video.append(frame)
    if generate_video:
      tools.generate_video_from_list_frame(list_frame=new_video,
                                          fps=FPS,
                                          path_video_output=path_video_output)
    else:
      return new_video,FPS

  
  # def generate_face_oval_video(self,path_video_input,path_video_output,align=False):
  #   routes_idx = self.FACE_OVAL
  #   new_video = []
  #   frame_list = self._get_list_frame(path_video_input,align=align)
  #   detection_result_list = self.extract_facial_landmarks(frame_list)
  #   start_time = time.time()
  #   # dict_max_dim = {'width': 0, 'height': 0}
  #   for (img,timestamp),frame_landmarks  in tqdm.tqdm(zip(frame_list, detection_result_list),total=len(frame_list), desc="Generating face oval video..."):
  #     landmarks = self.convert_from_numpy_to_NormalizedLandmark(np.array(frame_landmarks))
  #     routes = self._find_coords_point(routes_idx, landmarks, img)
  #     frame,_ = self._extract_face_oval_from_img(img, routes)
  #     landmarks = self.get_numpy_array(landmarks)
  #     top_left_corner = (int(np.min(landmarks[:, 0]*frame.shape[1])),
  #                       int(np.min(landmarks[:, 1]*frame.shape[0])))
  #     bottom_right_corner = (int(np.max(landmarks[:, 0]*frame.shape[1])),
  #                       int(np.max(landmarks[:, 1]*frame.shape[0])))
  #     # check top_left_corner and bottom_right_corner
  #     if top_left_corner[0] < 0 or top_left_corner[1] < 0 or bottom_right_corner[0] > frame.shape[1] or bottom_right_corner[1] > frame.shape[0]:
  #       top_left_corner = (max(0, top_left_corner[0]), max(0, top_left_corner[1]))
  #       bottom_right_corner = (min(frame.shape[1], bottom_right_corner[0]), min(frame.shape[0], bottom_right_corner[1]))
  #     if frame.size == 0:
  #       print(f'Frame is empty')
  #     frame = self.post_process_frontalized_img(frontalized_img=frame,
  #                                               top_left_corner=top_left_corner,
  #                                               bottom_right_corner=bottom_right_corner,
  #                                               landmarks=landmarks)
  #     if frame is None:
  #       print(f"No face oval extracted. {path_video_input} at timestamp {timestamp}ms")
  #       continue
  #     new_video.append(frame)
      
  #   tools.generate_video_from_list_frame(list_frame=new_video,
  #                                        path_video_output=path_video_output)
    
  def get_frames_annotated(self,path_video_input,align=False):

    frame_list = self._get_list_frame(path_video_input,align=align)
    detection_result_list = self.extract_facial_landmarks(frame_list)
    new_video = []
    for (img, _), detection_result in zip(frame_list, detection_result_list):
      landmarks = detection_result.face_landmarks[0]
      landmarks = self.center_wrt_nose(landmarks)
      annotated_img,_ = self.plot_landmarks(image=img,
                                          landmarks=landmarks,
                                          connections=FACE_TESSELATION)
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
    nose = landmarks[NOSE_INDEX]
    
    if isinstance(landmarks,np.ndarray) and np.min(landmarks[:,:2])>=0 and np.max(landmarks[:,:2])<=1: # normalized landmarks
      # set nose landmarks in [0.5,0.5,0.5]
      centered_landmarks = landmarks - [nose-[0.5,0.5,0.5]]
    else:
      raise ValueError("Invalid landmarks format. Must be a list of NormalizedLandmark objects or a numpy array.")

    return centered_landmarks
  
  def center_frame_wrt_nose(self,frame,landmarks):
    nose = landmarks[NOSE_INDEX]
    centered_nose = (nose.x - 0.5, nose.y - 0.5)
    shift_x = int(centered_nose[0] * frame.shape[1])
    shift_y = int(centered_nose[1] * frame.shape[0])
    print(f'shift_x: {shift_x}, shift_y: {shift_y}')
    return shift_x,shift_y
  
  def frontalized_video(self,video_path,ref_landmarks,interpolation_mod_chunk=None,only_landmarks_crop=False,align_before_front=False,log_path=None,time_logs=False,extra_landmark_smoothing=None,plot_debug=False,plot_every=30,plot_output_dir=None,preprocess=True,stabilize=True):
    """
    Frontalize every detected frame of a video and return the frontalized frames/landmarks.

    Args (debug-plotting related only):
      plot_debug:      If True, save a 2x2 debug figure every plot_every frames (frontalization path only).
      plot_every:      Interval between debug figures, in frames. Only used when plot_debug is True.
      plot_output_dir: Directory where debug PNGs are written. Falls back to 'z_debug_frontalization'.
      preprocess:      If True (default), crop every frame to a stable face ROI before
                       landmark extraction (see _get_list_frame); helps on small-face videos.
      stabilize:       If True (default), temporally stabilize the preprocessing ROI and use
                       one constant-size, smoothed-center output crop per video instead of a
                       per-frame landmark min/max box (see smooth_boxes). Removes the
                       frame-to-frame position/scale jitter of the output video. False
                       restores the old per-frame behavior.
    """

    def validate_frame_detection(list_to_validate):
      miss_detection = False
      list_is_detected = []
      for idx, el in enumerate(list_to_validate):
        if el is None:
          list_is_detected.append(False)
          miss_detection = True
        else:
          list_is_detected.append(True)
      return miss_detection, list_is_detected

    # start = time.time()
    list_frames,list_timestamp,fps = self._get_list_frame(video_path,align=align_before_front,return_tuple=False,preprocess=preprocess,stabilize=stabilize)
    if interpolation_mod_chunk is not None and len(interpolation_mod_chunk) != 2:
      raise ValueError(f'interpolation_mod_chunk must have len == 2, position 0 string for modality (spread_linearly or mirror_start_video), position 1 chunk size')
    if interpolation_mod_chunk is not None and interpolation_mod_chunk[0] == 'spread_linearly':
      # spread_linearly cannot pad landmarks, so it must pad frames before detection. Frames
      # later dropped by detection can then break the chunk-multiple guarantee; mirror_start_video
      # instead pads after detection filtering (below) and does not have this problem.
      list_frames,list_timestamp = self.apply_video_interpolation(frame_list=list_frames,
                                                                mod=interpolation_mod_chunk[0],
                                                                chunk_size=interpolation_mod_chunk[1],
                                                                fps=fps,
                                                                )
    # print("Time to get list frame: ",time.time()-start)
    tuple_frames_timestamp = list(zip(list_frames,list_timestamp))
    miss_detection, list_is_detected = validate_frame_detection(tuple_frames_timestamp)
    
    if miss_detection:
      raise DetectionError(f"No face detected in some frames during alignment in {video_path}", list_is_detected)
    else:
      list_landmarks = self.extract_facial_landmarks(tuple_frames_timestamp)
      miss_detection, list_is_detected = validate_frame_detection(list_landmarks)
      
      # Filter out frames where no face was detected
      if miss_detection:
        list_frames = [frame for frame, is_detected in zip(list_frames, list_is_detected) if is_detected]
        timestamp_list = [timestamp for timestamp, is_detected in zip(list_timestamp, list_is_detected) if is_detected]
        list_landmarks = [landmarks for landmarks, is_detected in zip(list_landmarks, list_is_detected) if is_detected]
        list_landmarks = np.array(list_landmarks)
        tuple_frames_timestamp = list(zip(list_frames,timestamp_list))
      else:
        list_landmarks = np.array(list_landmarks)
      
      if miss_detection and False: # disable for now
        raise DetectionError(f"No face detected in some frames during landmarks detection in {video_path}", list_is_detected)
      else:
        list_frontalized_img = []
        list_frontalized_landmarks = []
        list_frames = [frame for frame, _ in tuple_frames_timestamp]
        del tuple_frames_timestamp
        if interpolation_mod_chunk is not None and interpolation_mod_chunk[0] == 'mirror_start_video':
          # Pad AFTER detection filtering so the output frame count is guaranteed to be a
          # multiple of the chunk size; landmarks are padded with the same mirrored indices.
          list_frames, _, list_landmarks = self.apply_video_interpolation(frame_list=list_frames,
                                                                mod=interpolation_mod_chunk[0],
                                                                chunk_size=interpolation_mod_chunk[1],
                                                                fps=fps,
                                                                landmarks_list=list(list_landmarks))
          list_landmarks = np.array(list_landmarks)
        # print(f'Elapsed time to extract landmarks: {time.time()-start}')
        if extra_landmark_smoothing is not None and isinstance(extra_landmark_smoothing,LandmarkSmoother):
          print(f"Additional landmark smoothing: {extra_landmark_smoothing.method}")
          list_landmarks = extra_landmark_smoothing.smooth(list_landmarks)
        # start = time.time()
        if not only_landmarks_crop:
          # Pass 1 (cheap, transforms only): frontalize the landmarks of every frame
          # first so the output crop geometry can be derived for the whole video.
          for landmarks in list_landmarks:
            rotation, translation = self.compute_rigid_transform(landmarks, ref_landmarks)
            list_frontalized_landmarks.append(self.apply_rigid_transform(rotation, translation, landmarks).T)

          crop_boxes = None
          if stabilize:
            # One constant-size crop per video with a smoothed center: per-frame min/max
            # boxes vary in size, and the video writer then stretch-resizes every frame
            # to a common size, which shows up as scale/position pulsing. In frontalized
            # space the face is quasi-static (aligned to ref_landmarks), so a max-sized
            # box tracking the smoothed center cannot lose it. The 2 px pad absorbs
            # per-frame deviations from the smoothed center.
            crop_boxes = []
            for frame, frontalized_landmarks in zip(list_frames, list_frontalized_landmarks):
              h, w = frame.shape[:2]
              xs = frontalized_landmarks[:, 0] * w
              ys = frontalized_landmarks[:, 1] * h
              crop_boxes.append((int(np.min(xs)) - 2, int(np.min(ys)) - 2,
                                 int(np.max(xs)) + 2, int(np.max(ys)) + 2))
            crop_boxes = smooth_boxes(crop_boxes, list_frames[0].shape[:2])

          # Pass 2 (expensive): warp each frame and crop it with its precomputed box.
          for count, (frame, landmarks, frontalized_landmarks) in tqdm.tqdm(
              enumerate(zip(list_frames, list_landmarks, list_frontalized_landmarks)),
              total=len(list_frames), desc="Frontalizing frames..."):
            frontalized_img_SVD = self._get_frontalized_img(landmarks_2d=landmarks,
                                                            frontalized_landmarks_2d=frontalized_landmarks,
                                                            orig_frame=frame,
                                                            log_path=log_path,
                                                            nr_frame=count)

            if crop_boxes is not None:
              x0, y0, x1, y1 = crop_boxes[count]
              top_left_corner = (x0, y0)
              bottom_right_corner = (x1, y1)
            else:
              top_left_corner = (int(np.min(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                                int(np.min(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
              bottom_right_corner = (int(np.max(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                                int(np.max(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))

            frontalized_img_SVD = self.post_process_frontalized_img(frontalized_img=frontalized_img_SVD,
                                        top_left_corner=top_left_corner,
                                        bottom_right_corner=bottom_right_corner,
                                        landmarks=frontalized_landmarks,
                                        )
            if plot_debug and (count % plot_every == 0):
              video_name = os.path.splitext(os.path.basename(video_path))[0]
              save_path = os.path.join(plot_output_dir or 'z_debug_frontalization',
                                       f'{video_name}_frame{count}.png')
              self.plot_frontalization_debug(orig_frame=frame,
                                             frontalized_img=frontalized_img_SVD,
                                             original_landmarks=landmarks,
                                             frontalized_landmarks=frontalized_landmarks,
                                             save_path=save_path)
            list_frontalized_img.append(frontalized_img_SVD)
        else:
          for count, (frame, landmarks) in tqdm.tqdm(enumerate(zip(list_frames, list_landmarks)), total=len(list_frames), desc="Cropping frames to face oval..."):
            frame,mask = self.extract_frame_oval_from_img(frame,landmarks)
            if frame.size == 0:
              print(f'Frame is empty after extracting face oval at count {count}')
            top_left_corner = (int(np.min(landmarks[:, 0]*frame.shape[1])),
                              int(np.min(landmarks[:, 1]*frame.shape[0])))
            bottom_right_corner = (int(np.max(landmarks[:, 0]*frame.shape[1])),
                              int(np.max(landmarks[:, 1]*frame.shape[0])))
            # check top_left_corner and bottom_right_corner
            if top_left_corner[0] < 0 or top_left_corner[1] < 0 or bottom_right_corner[0] > frame.shape[1] or bottom_right_corner[1] > frame.shape[0]:
              top_left_corner = (max(0, top_left_corner[0]), max(0, top_left_corner[1]))
              bottom_right_corner = (min(frame.shape[1], bottom_right_corner[0]), min(frame.shape[0], bottom_right_corner[1]))
            if frame.size == 0:
              print(f'Frame is empty before cropping at count {count}')
            frame = self.post_process_frontalized_img(frontalized_img=frame,
                                                      top_left_corner=top_left_corner,
                                                      bottom_right_corner=bottom_right_corner,
                                                      landmarks=landmarks)
            list_frontalized_img.append(frame)
            # check if array is empty
            if frame.size == 0:
              print("Empty frame after cropping.")
            list_frontalized_landmarks.append(landmarks)
          print("dbe")
        return{
          'list_frontalized_frame': list_frontalized_img,
          'list_frontalized_landmarks': list_frontalized_landmarks,
          'list_is_detected': list_is_detected,
          'FPS': fps,
        } 

  def frontalize_img(self,frame,ref_landmarks,align=True,time_logs=False,v2=False,stop_after=-1,log_path=None):
    start = time.time()
    if align:
      orig_frame = np.array(self.align_face(frame),dtype=np.uint8)
    else:
      orig_frame = cv2.copyMakeBorder(frame,40,40,40,40,cv2.BORDER_CONSTANT,value=(0,0,0))
      cv2.imwrite('z_debug_frontalization/tmp.jpg',cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR))
      

    landmarks = self.extract_facial_landmarks([(orig_frame, 0)])
    print(f'landmarks: {landmarks}')
    if landmarks[0] is not None: # or len(landmarks[0].face_landmarks) > 0:
      # landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0].face_landmarks[0]])
      landmarks = landmarks.squeeze()
      # print(f'Frontalizing frame {count}/{len(list_frames)}')
      rotation, translation = self.compute_rigid_transform(landmarks, ref_landmarks)
      frontalized_landmarks = self.apply_rigid_transform(rotation, translation, landmarks).T

      
      frontalized_img_SVD = self._get_frontalized_img(landmarks_2d=landmarks,
                                                      frontalized_landmarks_2d=frontalized_landmarks,
                                                      orig_frame=frame,
                                                      log_path=log_path,
                                                      nr_frame=0)

      top_left_corner = (int(np.min(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                        int(np.min(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))
      bottom_right_corner = (int(np.max(frontalized_landmarks[:, 0]*frontalized_img_SVD.shape[1])),
                        int(np.max(frontalized_landmarks[:, 1]*frontalized_img_SVD.shape[0])))

      frontalized_img_SVD = self.post_process_frontalized_img(frontalized_img=frontalized_img_SVD,
                                                              top_left_corner=top_left_corner,
                                                              bottom_right_corner=bottom_right_corner,
                                                              landmarks=frontalized_landmarks,
                                                              )
      return{
        'frontalized_img': frontalized_img_SVD,
        # 'frontalized_norm_landmarks': rot_trans_landmarks,
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

  def post_process_frontalized_img(self,frontalized_img,top_left_corner,bottom_right_corner,landmarks,apply_mirroring_reconstruction=False):
    if apply_mirroring_reconstruction:
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
    # frontalized_img=cv2.resize(frontalized_img,(224,224))

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

  def apply_delaunay_triangulation_v2(self, original_image, frontalized_landmarks, original_landmarks, stop_after=-1,log_path=None,nr_frame=None):
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
      start_row = y_orig_rect - delta
      end_row = y_orig_rect + h_orig_rect + delta
      start_col = x_orig_rect - delta
      end_col = x_orig_rect + w_orig_rect + delta
      if end_row > h_orig or end_col > w_orig or start_row < 0 or start_col < 0:
        bottom_border = max(0,end_row - h_orig)
        right_border = max(0,end_col - w_orig)
        left_border = abs(min(start_col,0))
        top_border = abs(min(start_row,0))
        bordered_image = cv2.copyMakeBorder(original_image,top_border,bottom_border,left_border,right_border,cv2.BORDER_CONSTANT,value=(0,0,0))
        img_orig_cut = bordered_image[start_row+top_border:end_row+top_border, start_col+left_border:end_col+left_border]
      else:
        img_orig_cut = original_image[start_row:end_row, start_col:end_col]
      affine_transform_norm = cv2.getAffineTransform(np.array(normalized_original_triangle,dtype=np.float32),
                                                   np.array(normalized_frontalized_triangle,dtype=np.float32))
      # try:
      wrp_region = cv2.warpAffine(img_orig_cut, affine_transform_norm, (w_front+delta, h_front+delta))
      # except Exception as e:
      #   print(f'Error in cv2.warpAffine: {e}')
      #   print(f'img_orig_cut shape: {img_orig_cut.shape}')
      #   print(f'frontalized_triangle: {frontalized_triangle}')
      #   print(f'original_triangle: {original_triangle}')
      #   print(f'normalized_frontalized_triangle: {normalized_frontalized_triangle}')
      #   print(f'normalized_original_triangle: {normalized_original_triangle}')
      #   raise e
      wrp_region = wrp_region[delta:delta+h_front,delta:delta+w_front]
      
      rect_mask = np.zeros((h_front, w_front), dtype=np.uint8)
      # Create a mask for the current triangle
      triangle_in_rect = frontalized_triangle - [x_front, y_front]
      cv2.fillConvexPoly(rect_mask, np.int32(triangle_in_rect), (255))

      mask_expanded = rect_mask[:, :, np.newaxis]
      # Clip the destination rectangle to the canvas: the rigid transform can push
      # frontalized landmarks (slightly) outside the frame, giving a bounding rect with
      # negative x/y or extending past the bottom/right edge.
      dst_y0, dst_y1 = max(0, y_front), min(frontalized_image.shape[0], y_front + h_front)
      dst_x0, dst_x1 = max(0, x_front), min(frontalized_image.shape[1], x_front + w_front)
      if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        continue
      mask_expanded = mask_expanded[dst_y0 - y_front:dst_y1 - y_front,
                                    dst_x0 - x_front:dst_x1 - x_front]
      wrp_region = wrp_region[dst_y0 - y_front:dst_y1 - y_front,
                              dst_x0 - x_front:dst_x1 - x_front]
      # adapt the mask to the frontalized image
      frontalized_image[dst_y0:dst_y1, dst_x0:dst_x1] = (
        wrp_region * (mask_expanded / 255.0) +
        frontalized_image[dst_y0:dst_y1, dst_x0:dst_x1] * (1 - mask_expanded / 255.0)
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

  def _get_frontalized_img(self,frontalized_landmarks_2d, landmarks_2d, orig_frame, v2=False,stop_after=-1,log_path=None,nr_frame=None):
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
                                                log_path=log_path,
                                                nr_frame=nr_frame)
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

    top_left_corner = (np.min(landmarks[:, 0])-padding, np.min(landmarks[:, 1])-padding)
    bottom_right_corner = (np.max(landmarks[:, 0])+padding, np.max(landmarks[:, 1])+padding)

    return img,top_left_corner,bottom_right_corner
    # return img

  def plot_frontalization_debug(self, orig_frame, frontalized_img, original_landmarks, frontalized_landmarks, save_path):
    """
    Save a 2x2 debug figure summarizing one frame of the frontalization process.

    Args:
      orig_frame:            Original RGB frame before frontalization. Shape: (H, W, 3).
      frontalized_img:       Final cropped frontalized image. Shape: (H', W', 3).
      original_landmarks:    Detected landmarks before frontalization. Shape: (N, 2) or (N, 3), normalized.
      frontalized_landmarks: Landmarks after the rigid frontalization transform. Shape: (N, 2) or (N, 3), normalized.
      save_path:             Full path (including .png filename) where the figure is written.

    Returns:
      None. Writes the figure to save_path as a side effect.
    """
    def _mesh_crop(landmarks):
      mesh, top_left, bottom_right = self.plot_landmarks_triangulation(
        image=np.zeros_like(orig_frame), landmarks=np.asarray(landmarks), padding=10)
      # corners can go negative (min - padding); clamp so numpy slicing does not wrap
      top_left = (max(0, int(top_left[0])), max(0, int(top_left[1])))
      bottom_right = (max(0, int(bottom_right[0])), max(0, int(bottom_right[1])))
      return mesh[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    mesh_before = _mesh_crop(original_landmarks)
    mesh_after = _mesh_crop(frontalized_landmarks)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].set_title('Landmarks before frontalization')
    ax[0, 0].imshow(mesh_before)
    ax[0, 1].set_title('Landmarks after frontalization')
    ax[0, 1].imshow(mesh_after)
    ax[1, 0].set_title('Original image')
    ax[1, 0].imshow(orig_frame)
    ax[1, 1].set_title('Frontalized image')
    ax[1, 1].imshow(frontalized_img)
    for a in ax.ravel():
      a.axis('off')

    save_dir = os.path.dirname(save_path)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    # thight layout to avoid cutting off titles
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved frontalization debug figure in {save_path}')


class LandmarkSmoother:
  def __init__(self, method="kalman", window_size=5):
    """
    Initializes the faceExtractor with the specified method and window size.

    Args:
      method (str): The method to be used for smoothing. Can be "savgol", "moving_average",
                    "median_filter", or "kalman". "savgol" is zero-phase (no temporal lag)
                    and preserves fast expression onsets better than a trailing average.
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
    landmarks = np.array(landmarks)  # Shape: (num_frames, num_landmarks, 2 or 3)
    num_frames, num_points, _ = landmarks.shape

    if self.method == "savgol":
      return self.savgol(landmarks)
    elif self.method == "moving_average":
      return self.moving_average(landmarks)
    elif self.method == "median_filter":
      return self.median_filter(landmarks)
    elif self.method == "kalman":
      if not self.kalman_filters:
        self.initialize_kalman_filters(num_points)
      return self.kalman_filter_smoothing(landmarks)
    else:
      raise ValueError("Invalid smoothing method!")

  def savgol(self, landmarks):
    """
    Zero-phase Savitzky-Golay filtering along the time axis.

    Args:
      landmarks: Landmark trajectories. Shape: (num_frames, num_landmarks, 2 or 3).

    Returns:
      Smoothed landmarks, same shape. Videos shorter than the minimum valid window
      (5 frames) are returned unchanged.
    """
    num_frames = landmarks.shape[0]
    window = min(self.window_size, num_frames)
    if window % 2 == 0:
      window -= 1
    if window < 5:  # must exceed polyorder=2; below 5 the fit is (near) identity
      return landmarks
    return savgol_filter(landmarks, window, polyorder=2, axis=0)

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
        # only x/y are filtered; extra coordinates (z) pass through untouched
        smoothed[frame_idx, point_idx, :2] = prediction[:2].flatten()
        smoothed[frame_idx, point_idx, 2:] = landmarks[frame_idx, point_idx, 2:]
    return smoothed

class DetectionError(Exception):
  def __init__(self, message, list_no_detection_idx):
    super().__init__(message)
    self.list_no_detection_idx = list_no_detection_idx
