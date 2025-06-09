import sys
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array
from pathlib import Path


def add_title(clip, title, title_height=50, fontsize=24, font='Arial', color='white', bg_color='black'):
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

def main(video1_path, video2_path, output_path="comparison.mp4"):
  # Load the video clips
  clip1 = VideoFileClip(video1_path)
  clip2 = VideoFileClip(video2_path)

  # Create composite clips with titles above each video
  clip1_with_title = add_title(clip1, Path(video1_path).parts[-3][-4:])
  clip2_with_title = add_title(clip2, Path(video2_path).parts[-3][-4:])

  # Ensure both clips have the same height for side-by-side display.
  # Optionally, you can resize if necessary. Here we assume they have similar dimensions.
  final_clip = clips_array([[clip1_with_title, clip2_with_title]])

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
  video1_path = "partA/video/video_frontalized/082414_m_64/082414_m_64-BL1-088.mp4"
  video2_path = "partA/video/video_frontalized_new_stab/082414_m_64/082414_m_64-BL1-088.mp4"
  output_path = f'comparison_{video1_path.split("/")[-1].split(".")[0]}_{video2_path.split("/")[-1].split(".")[0]}.mp4'
  main(video1_path, video2_path, output_path)
