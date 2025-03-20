import os
import custom.tools as tools
import argparse


def generate_path(path):
  if path[0] == '/':
    return path
  return os.path.join('/equilibrium','fvilli','PainAssessmentVideo',path)

def main(csv_path,video_path,log_folder_path):
  csv_name = os.path.split(csv_path)[1]
  csv_array,_ = tools.get_array_from_csv(csv_path=csv_path)
  list_csv_sample = csv_array[:,-1]
  list_video_folder = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path,name))]
  list_video_name = []
  for folder in list_video_folder:
    sample_folder = os.path.join(video_path,folder)
    list_video_name += [f.split('.mp4')[0] for f in os.listdir(sample_folder) if f.endswith('.mp4')]
  print(f'list_video_name: {len(list_video_name)}')
  # list_video_name = [f.split('.mp4')[0] for f in list_video_name]
  count = 0
  log_file = os.path.join(log_folder_path,f'log_video_{csv_name[:-4]}.txt')
  if os.path.exists(log_file):
    os.remove(os.path.join(log_file))
  count_pos = 0
  print(list_csv_sample)
  for sample in list_csv_sample:
    if sample not in list_video_name:
      print(f'{sample} not in {video_path}')
      count += 1
      with open(log_file,'a') as f:
        f.write(f'{sample}.mp4\n')
    # else:
    #   count_pos += 1
    #   list_video_name.remove(sample)
      # print(f'len: {len(list_video_name)}')
      #
  print(f'Number of video in the folder: {len(list_video_name)}')
  print(f'Number of video in the csv   : {len(list_csv_sample)}')
  print(f'Number of video not in folder: {count}')
  print(f'logs saved in {log_file}')
  # print(f'csv name: {}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.description = 'Check if all videos in csv are in the video folder'
  parser.add_argument('--csv_path', type=str, required=True,help='Path to csv file')
  parser.add_argument('--f_video_path', type=str, required=True,help='Path to folder video')
  parser.add_argument('--g_path',action='store_true', help='Add /equilibrium/fvilli/PainAssessmentVideo to all paths')
  parser.add_argument('--log_folder_path', type=str, default=os.path.join('partA','video','logs_video'), help='Path to log folder')
  args = parser.parse_args()
  if args.g_path:
    args.csv_path = generate_path(args.csv_path)
    args.video_path = generate_path(args.f_video_path)
    args.log_folder_path = generate_path(args.log_folder_path)
    print(f'csv path: {args.csv_path}\n')
    print(f'video path: {args.f_video_path}')
  if not os.path.exists(args.log_folder_path):
    os.makedirs(args.log_folder_path)
  print(f'log folder path: {args.log_folder_path}\n')
  
  main(csv_path=args.csv_path,
       video_path=args.f_video_path,
       log_folder_path=args.log_folder_path)