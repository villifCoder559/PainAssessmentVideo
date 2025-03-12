import pickle
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
  pkl_path = os.path.join('partA','video','roll_pitch_yaw_per_subject','roll_pitch_yaw_per_subject_all')
  dict_results = pickle.load(open(pkl_path,'rb'))
  list_record = []
  for sample_id,dict_angles in dict_results.items():
    abs_dict_angles = {}
    for k,v in dict_angles.items():
      for angle,freq in v.items():
        abs_dict_angles[k] = abs_dict_angles.get(k,{})
        abs_dict_angles[k][abs(angle)] = abs_dict_angles[k].get(abs(angle),0) + freq
    tot_abs_frames = sum(abs_dict_angles['roll'].values())
    tot_frames = sum(dict_angles['roll'].values())
    if tot_abs_frames != tot_frames:
      print(f'Error: tot_abs_frames {tot_abs_frames} != tot_frames {tot_frames}')
    max_roll = max(abs_dict_angles['roll'].keys())
    max_pitch = max(abs_dict_angles['pitch'].keys())
    max_yaw = max(abs_dict_angles['yaw'].keys())
    frequency_max_roll = abs_dict_angles['roll'][max_roll]
    frequency_max_pitch = abs_dict_angles['pitch'][max_pitch]
    frequency_max_yaw = abs_dict_angles['yaw'][max_yaw]
    roll_0_3 = np.abs(list(abs_dict_angles['roll'].keys())) < 3
    roll_3_6 = (np.abs(list(abs_dict_angles['roll'].keys())) >= 3) * (np.abs(list(abs_dict_angles['roll'].keys())) < 6)
    roll_6_inf = np.abs(list(abs_dict_angles['roll'].keys())) >= 6
    pitch_0_3 = np.abs(list(abs_dict_angles['pitch'].keys())) < 3
    pitch_3_6 = (np.abs(list(abs_dict_angles['pitch'].keys())) >= 3) * (np.abs(list(abs_dict_angles['pitch'].keys())) < 6)
    pitch_6_inf = np.abs(list(abs_dict_angles['pitch'].keys())) >= 6
    yaw_0_3 = np.abs(list(abs_dict_angles['yaw'].keys())) < 3
    
    list_record.append({
      'sample': sample_id,
      'subject_name': sample_id.split('_')[0],
      'max_roll': max_roll,
      'frequency_max_roll': frequency_max_roll,
      'max_pitch': max_pitch,
      'frequency_max_pitch': frequency_max_pitch,
      'max_yaw': max_yaw,
      'frequency_max_yaw': frequency_max_yaw,
      'count_roll_0_3': np.sum(np.array(list(abs_dict_angles['roll'].values()))[roll_0_3]),
      'count_roll_3_6': np.sum(np.array(list(abs_dict_angles['roll'].values()))[roll_3_6]),
      'count_roll_6_inf': np.sum(np.array(list(abs_dict_angles['roll'].values()))[roll_6_inf]),
      'count_pitch_0_3': np.sum(np.array(list(abs_dict_angles['pitch'].values()))[pitch_0_3]),
      'count_pitch_3_6': np.sum(np.array(list(abs_dict_angles['pitch'].values()))[pitch_3_6]),
      'count_pitch_6_inf': np.sum(np.array(list(abs_dict_angles['pitch'].values()))[pitch_6_inf]),
      'count_yaw_0_3': np.sum(np.array(list(abs_dict_angles['yaw'].values()))[yaw_0_3]),
      'tot_frames': sum(abs_dict_angles['roll'].values())
    })
  df = pd.DataFrame(list_record)
  df.to_csv(os.path.join('partA','video','roll_pitch_yaw_per_subject','roll_pitch_yaw_per_subject_all.csv'),index=False)
  print('All results saved in partA/video/roll_pitch_yaw_per_subject_all.csv')