import cv2
import av
import time

# Video file path
# video_path = "your_video_file.mp4"
video_path = 'partA/video/video/112016_m_25/112016_m_25-BL1-081.mp4'

# OpenCV reading speed
def opencv_read_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    cap.release()
    end_time = time.time()
    print(f"OpenCV: Read {frame_count} frames in {end_time - start_time:.2f} seconds.")

# av (FFmpeg) reading speed
def av_read_speed(video_path):
    container = av.open(video_path)
    frame_count = 0
    start_time = time.time()
    
    for frame in container.decode(video=0):
        frame_count += 1
    
    end_time = time.time()
    print(f"av: Read {frame_count} frames in {end_time - start_time:.2f} seconds.")
    # av (FFmpeg) reading speed with GPU enabled
def av_read_speed_gpu(video_path):
  container = av.open(video_path)
  stream = container.streams.video[0]
  stream.codec_context.options = {'hwaccel': 'cuda'}
  frame_count = 0
  start_time = time.time()
  
  for frame in container.decode(video=0):
    frame_count += 1
  
  end_time = time.time()
  print(f"av (GPU): Read {frame_count} frames in {end_time - start_time:.2f} seconds.")
# Run the comparison

opencv_read_speed(video_path)
av_read_speed(video_path)
# av_read_speed_gpu(video_path)
