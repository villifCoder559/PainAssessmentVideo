import cv2
import mediapipe as mp
import numpy as np
import os

import cv2
import mediapipe as mp
import numpy as np

def compute_landmark_movements(video_path):
  """
  Computes the average movement of facial landmarks between consecutive frames in a video.

  Args:
    video_path (str): Path to the input video file.

  Returns:
    movements (list): List of average landmark movements between consecutive frames.
  """
  # Initialize MediaPipe Face Mesh
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

  # Open the video file
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    return []

  movements = []
  prev_landmarks = None

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame_bordered = np.zeros((rgb_frame.shape[0]*2,rgb_frame.shape[1]*2,3), np.uint8)
    rgb_frame_bordered[rgb_frame.shape[0]//2:rgb_frame.shape[0]*3//2,rgb_frame.shape[1]//2:rgb_frame.shape[1]*3//2] = rgb_frame
    rgb_frame = rgb_frame_bordered
    # show the frame
    
    cv2.imshow('frame',cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    # Process the frame to detect facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
      # Extract the landmarks
      landmarks = np.array([(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark])

      if prev_landmarks is not None:
        # Compute the Euclidean distance between corresponding landmarks
        distances = np.linalg.norm(landmarks - prev_landmarks, axis=1)
        # Calculate the average movement
        avg_movement = np.mean(distances)
        movements.append(avg_movement)

      # Update the previous landmarks
      prev_landmarks = landmarks

  cap.release()
  return movements

# Example usage
video_path = os.path.join('partA','video','video_frontalized','082414_m_64','082414_m_64-BL1-083.mp4')
movements = compute_landmark_movements(video_path)
if movements:
  print(f"Average landmark movement per frame: {np.mean(movements):.4f} units")
else:
  print("No movements detected or unable to process the video.")


def old():
  # Initialize MediaPipe Face Mesh
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.1)
  face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

  # Paths to videos
  video_path_orig = '/media/villi/TOSHIBA EXT/orig_video/video/video/082414_m_64/082414_m_64-BL1-083.mp4'
  video_path_front = os.path.join('partA','video','video_frontalized','082414_m_64','082414_m_64-BL1-083.mp4')

  # Open videos
  cap1 = cv2.VideoCapture(video_path_orig)
  cap2 = cv2.VideoCapture(video_path_front)

  def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0  # Returns True if a face is detected

  # Function to get facial landmarks
  def get_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
      landmarks = np.array([(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark])  # Normalize coordinates
      # nose = landmarks[1]
      # centered_landmarks = landmarks - nose
      return landmarks
    return None

  # Read first frame
  ret1, prev_frame1 = cap1.read()
  ret2, prev_frame2 = cap2.read()

  if not ret1 or not ret2:
    print("Error: Unable to read one or both videos.")
    cap1.release()
    cap2.release()
    exit()
  # add bloack borders to prev_frame2
  prev_landmarks1 = get_landmarks(prev_frame1)
  prev_landmarks2 = get_landmarks(prev_frame2)

  frame_index = 0

  while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    bordered_frame_2 = np.zeros((frame2.shape[0]*2,frame2.shape[1]*2,3), np.uint8)
    bordered_frame_2[frame2.shape[0]//2:frame2.shape[0]*3//2,frame2.shape[1]//2:frame2.shape[1]*3//2] = frame2
    frame2 = bordered_frame_2

    if not ret1 or not ret2:
      break

    landmarks1 = get_landmarks(frame1)
    landmarks2 = get_landmarks(frame2)
    face_cascade_detected1 = detect_face(frame1)
    face_cascade_detected2 = detect_face(frame2)
    # Visualize frames
    # cv2.imshow("Original", frame1)
    # cv2.imshow("Frontalized", frame2)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    if landmarks1 is not None and landmarks2 is not None and prev_landmarks1 is not None and prev_landmarks2 is not None:
      # Compute landmark displacement (Euclidean distance)
      movement1 = np.linalg.norm(landmarks1 - prev_landmarks1, axis=1).mean()
      movement2 = np.linalg.norm(landmarks2 - prev_landmarks2, axis=1).mean()

      # Compute the difference in movement between the original and frontalized videos
      movement_diff = movement2 - movement1
      print(f"Frame {frame_index}: Landmark Movement Difference = {movement_diff:.4f}")

    # Update previous landmarks
    prev_landmarks1 = landmarks1
    prev_landmarks2 = landmarks2
    frame_index += 1

  cap1.release()
  cap2.release()
