import pandas
import os 


def convert_txt_to_csv(input_file, output_folder,dataset_folder=None):
  with open(input_file, 'r') as file:
    lines = file.readlines()
  
  label_to_class = {
    'boxing': 0,
    'handclapping': 1,
    'handwaving': 2,
    'jogging': 3,
    'running': 4,
    'walking': 5,
  }
  data = []
  count_sample = 0
  for line in lines:
    # remove tabs
    line = line.replace('\t', ' ')
    split_line = (line.strip().split(' ')[0]).split('_') # person{X}_{action}_{type}
    if len(split_line) < 2:
      print("Empty line found, skipping...")
      continue
    data.append({
      'subject_id': split_line[0][-2:].strip(),
      'subject_name': split_line[0].strip(),
      'class_id': label_to_class[split_line[1]],
      'class_name': split_line[1].strip(),
      'sample_id':count_sample,
      'sample_name': "_".join(split_line[:3]).strip(), 
    })
    count_sample += 1
    subject_folder = os.path.join(dataset_folder, split_line[0])
    if data is not None and not os.path.exists(subject_folder):
      os.makedirs(subject_folder)
      print(f"Created folder: {subject_folder}")
  df = pandas.DataFrame(data)
  output_file = os.path.join(output_folder, 'kth_samples.csv')
  df.to_csv(output_file, index=False,sep='\t')
  print(f"CSV file saved to {output_file}")
  
  
  
if __name__ == "__main__":
  input_file = 'KTH_dataset/starting_point/dataset_description.txt'
  output_folder = 'KTH_dataset/starting_point'
  dataset_folder = 'KTH_dataset'
  convert_txt_to_csv(input_file, output_folder, dataset_folder)