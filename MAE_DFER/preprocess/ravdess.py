# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys
import glob

dataset = 'RAVDESS'
data_path = '/path/to/ravdess'
video_dir = os.path.join(data_path, 'face_aligned') # Extract face from original video by OpenFace (https://github.com/TadasBaltrusaitis/OpenFace)

# specify the split/fold id: 1, 2, ..., 6
assert len(sys.argv) > 1, 'Please input the split/fold id (1-6):'
split = int(sys.argv[1])
assert 1 <= split <=6, 'Error: split/fold id must be in (1-6)!'

save_dir = f'../saved/data/ravdess/split0{split}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read
sample_dirs = sorted(glob.glob(os.path.join(video_dir, '*/*')))
assert len(sample_dirs) == 1440, f'Error: wrong number of samples, expected 1440, got {len(sample_dirs)}.'
train_label_dict, test_label_dict = {}, {}
for sample_dir in sample_dirs:
    sub_id, sample_id = sample_dir.split('/')[-2:]
    sub_idx = int(sub_id.split('_')[-1]) - 1
    label_idx = int(sample_id.split('-')[2]) - 1
    if (split-1)*4 <= sub_idx < split*4:
        test_label_dict[sample_dir] = label_idx
    else:
        train_label_dict[sample_dir] = label_idx
print(f'Total samples: {len(sample_dirs)}, train={len(train_label_dict)}, test={len(test_label_dict)}')

# write
new_train_split_file = os.path.join(save_dir, f'train.csv')
df = pd.DataFrame(train_label_dict.items())
df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

new_test_split_file = os.path.join(save_dir, f'test.csv')
df = pd.DataFrame(test_label_dict.items())
df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

## val == test, simply specify in the code, do not generate the csv file
