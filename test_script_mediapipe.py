import mediapipe as mp
from PIL import Image
import numpy as np
import cv2

# from google.colab.patches import cv2_imshow
# Initialize FaceAligner
print('Initialize FaceAligner')
mp_face_aligner = mp.tasks.vision.FaceAligner.create_from_model_path(
    model_path="landmark_model/face_landmarker.task"
)

def align_face_mediapipe(image_path):
  # Load image
  image = cv2.imread(image_path)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
  print(f'image shape numpy_view(): {mp_image.numpy_view().shape}')
  # Perform alignment
  print("Before align")
  aligned_image = mp_face_aligner.align(mp_image)
  print("After align")
  if aligned_image is None:
      print("No face detected.")
      return image
  
  # Convert back to OpenCV format for display
  print("Before  get img")
  aligned_image_np = aligned_image.numpy_view()
  print("After get img")
  return aligned_image_np

# Test the FaceAligner
print('Test the FaceAligner')
print('numpy version:', np.__version__)
image_path = "images.jpeg"  # Replace with your image path
img=cv2.imread(image_path)
print(f'Original image dimension {img.shape}')
aligned_image = align_face_mediapipe(image_path)

# cv2.imshow('title',img)
# cv2.imshow('title2',aligned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(f'ALigned image dimension {aligned_image.shape}')
print(type(img))