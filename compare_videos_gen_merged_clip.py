import sys
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array
from pathlib import Path
import os
from moviepy.editor import ColorClip
import custom.tools as tools
import pandas as pd


def add_title(clip, title, title_height=50, fontsize=12, font='Arial', color='white', bg_color='black'):
  """
  Creates a composite clip with a title above the given video clip.
  """
  # Create a text clip as the title with the same width as the video
  text_clip = TextClip(title, fontsize=fontsize, font=font, color=color,
                         bg_color=bg_color, size=(clip.w, title_height))
  text_clip = text_clip.set_duration(clip.duration)
  
  # Position the video clip just below the text clip
  composite = CompositeVideoClip([text_clip, clip.set_position(("center", title_height))],
                                 size=(clip.w, clip.h + title_height))
  return composite

def main(video1_path, video2_path, output_path="comparison.mp4",std_size=(224,224)):
  # Load the video clips
  clip1 = VideoFileClip(video1_path)
  clip2 = VideoFileClip(video2_path)
  if clip1.size != std_size:
    clip1 = clip1.resize(std_size)
  if clip2.size != std_size:
    clip2 = clip2.resize(std_size)
  # Create composite clips with titles above each video
  clip1_with_title = add_title(clip1, Path(video1_path).parts[-3])
  clip2_with_title = add_title(clip2, Path(video2_path).parts[-3])
  padding_width = 10  # Width of the padding in pixels
  padding_height = clip1_with_title.h  # Same height as the clips with titles
  padding_clip = ColorClip(size=(padding_width, padding_height), color=(0, 0, 0),duration=clip1.duration)  # White padding
  
  # Ensure both clips have the same height for side-by-side display.
  # Optionally, you can resize if necessary. Here we assume they have similar dimensions.
  final_clip = clips_array([[clip1_with_title,padding_clip, clip2_with_title]])

  # Write the merged output to a file
  final_clip.write_videofile(output_path, codec="libx264")

if __name__ == "__main__":
  # if len(sys.argv) < 3:
  #   print("Usage: python compare_videos.py video1.mp4 video2.mp4 [output.mp4]")
  # else:
  #   video1_path = sys.argv[1]
  #   video2_path = sys.argv[2]
  #   output_path = sys.argv[3] if len(sys.argv) > 3 else "comparison.mp4"
  #   main(video1_path, video2_path, output_path)
  df = pd.read_csv('partA/starting_point/samples_exc_no_detection.csv',sep='\t')
  list_subject_id = df['subject_id'].unique()
  for subject_id in list_subject_id:
    subject_records = df[df['subject_id']==subject_id]
    record_target = subject_records.iloc[0]
    print(record_target)
    video1_path = f"partA/video/video_frontalized/{record_target['subject_name']}/{record_target['sample_name']}.mp4"
    video2_path = f"partA/video/video_frontalized_new/{record_target['subject_name']}/{record_target['sample_name']}.mp4"
    output_path = os.path.join('video_comparisons',f'comparison_{video1_path.split("/")[-1].split(".")[0]}_{video2_path.split("/")[-1].split(".")[0]}.mp4')
    main(video1_path, video2_path, output_path)
