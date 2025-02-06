import os
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import csv
import pandas as pd
from collections.abc import MutableMapping

def flatten_dict(d, parent_key='', sep='.'):
  """Recursively flattens a nested dictionary."""
  items = []
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, MutableMapping):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)

def save_dict_to_csv(data, saving_path):
  """Converts a list of nested dictionaries to a CSV file."""
  flattened_data = [flatten_dict(d) for d in data]
  df = pd.DataFrame(flattened_data).fillna('None')
  if not os.path.exists(os.path.dirname(saving_path)):
    os.makedirs(os.path.dirname(saving_path))
  df.to_csv(saving_path, index=False)

def filter_csv(filename, filter_conditions):
  """Reads the CSV and filters rows based on given column conditions.
  filter_conditions should be a dict where keys are column names and values are the values to match.
  """
  df = pd.read_csv(filename)
  for col, value in filter_conditions.items():
    df = df[df[col] == value]
  return df

def create_filtered_symlinks(csv_output, filter_dict, output_folder):
  """Creates symbolic links for images that match the filter conditions."""
  df = pd.read_csv(csv_output)
  
  # Apply filtering
  for col, value in filter_dict.items():
    df = df[df[col] == value]
  
  if df.empty:
    print("No matching entries found.")
    return
  
  # If output folder exists, delete it
  if os.path.exists(output_folder):
    for f in os.listdir(output_folder):
      if not f.endswith('.json'):
        os.remove(os.path.join(output_folder, f))
    
  os.makedirs(output_folder, exist_ok=True)
  all_images_folder = os.path.join('test_loss', "all_images")
  list_image = [f for f in os.listdir(all_images_folder) if os.path.isfile(os.path.join(all_images_folder, f))]
  
  if not list_image:
    print(f"No images found in {all_images_folder}")
    return
  
  # Create symbolic links
  for _, row in df.iterrows():
    image_id_folder = row.get('test_folder')
    
    # Filter the images in the 'all_images' folder that match the current 'test_folder'
    list_image_filtered = [os.path.join(all_images_folder, img_id) for img_id in list_image if image_id_folder in img_id]
    
    if not list_image_filtered:
      print(f"No images found for test folder: {image_id_folder}")
      continue

    for image_path in list_image_filtered:
      # Define symbolic link path
      link_name = os.path.join(output_folder, Path(image_path).name)
      if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
      # Create symbolic link if it doesn't exist
      try:
        if not os.path.exists(link_name):
          os.symlink(os.path.join(os.getcwd(),image_path), link_name)
          print(f"Created symlink: {link_name} -> {image_path}")
          list_image.remove(os.path.split(image_path)[-1])
        else:
          print(f"Symlink already exists: {link_name}")
      except OSError as e:
        print(f"Error creating symlink for {image_path}: {e}")
        
              
def find_pngs_and_config(test_path):
  image_paths = []
  config_path = None
  config_advanced_path = None
  for root, _, files in os.walk(test_path):
    if "global_config.json" in files:
      config_path = os.path.join(root, "global_config.json")
    if "advanced_model_config.json" in files:
      config_advanced_path = os.path.join(root, "advanced_model_config.json")
    image_paths.extend(os.path.join(root, f) for f in files if f == "train_test_loss.png" or f == "train_val_loss.png")
  return image_paths, config_path, config_advanced_path

def process_test_images(test_root_folder, output_folder,csv_output):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  list_dict = []
  for test_folder in os.listdir(test_root_folder):
    list_test_path = os.listdir(os.path.join(test_root_folder, test_folder))
    list_test_path = [os.path.join(test_root_folder, test_folder, f) for f in list_test_path]
    print(f"Processing: {test_folder}")
    for test_path in list_test_path:
      if not os.path.isdir(test_path):
        continue
      
      image_paths, config_path,config_advanced_path = find_pngs_and_config(test_path)
      if not image_paths or not config_path or not config_advanced_path:
        continue
      
      # Read config file
      with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
      with open(config_advanced_path, 'r', encoding='utf-8') as f:
        config_advanced_path = json.load(f)
      keys_to_print = ['k_fold','batch_size_training','epochs',
                      'criterion','optimizer_fn','lr','round_output_loss',
                      'init_network','regularization_lambda',
                      'regularization_loss','head_params',]
      config_data = {k: config_data[k] for k in keys_to_print if k in config_data}
      if 'head_params' not in config_data:
        config_data['head_params'] = config_advanced_path['head']
      # config_data['path_png'] = image_paths[0]
      config_text = json.dumps(config_data, indent=2)  # Format JSON nicely
      config_data['test_folder'] = Path(image_paths[0]).parts[-6]
      list_dict.append(config_data)
      # save_dict_to_csv([config_data], csv_output)
      
      # Define font
      try:
        font = ImageFont.truetype("Ubuntu-R.ttf", 20)
      except IOError:
        font = ImageFont.load_default()
      
      for idx, image_path in enumerate(image_paths):
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        # Create new image with additional space for text
        new_width = width + 600  # Extend width to accommodate text
        new_height = height + 200 # Extend height to accommodate text
        new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_img.paste(img, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(new_img)
        text_x = width + 10  # Position text after the image
        text_y = 10
        
        for line in config_text.split('\n'):
          draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
          text_y += 28  # Line spacing
        text_path_x = 10
        text_path_y = height + 50
        draw.text((text_path_x, text_path_y), image_path, fill=(0, 0, 0), font=font)
        # Save new image
        image_path_split = Path(image_path).parts
        output_image_name = f"{image_path_split[-6]}_{image_path_split[-3]}.png"
        output_image_path = os.path.join(output_folder,'all_images', output_image_name)
        if not os.path.exists(os.path.dirname(output_image_path)):
          os.makedirs(os.path.dirname(output_image_path))
        new_img.save(output_image_path)
        # print(f"Processed and saved: {output_image_path}")
  save_dict_to_csv(list_dict, csv_output)
  
# Example usage
generate = False
test_root_folder = "Tests"
output_folder = "test_loss"
csv_output = os.path.join(output_folder,"_all_configs.csv")
  
if generate:
  if os.path.exists(csv_output):
    os.remove(csv_output)
  process_test_images(test_root_folder, output_folder, csv_output)
else:
  # k_fold, batch_size_training, epochs, criterion,optimizer_fn, lr, round_output_loss, 
  # init_network,regularization_lambda,regularization_loss,head_params.input_size,
  # head_params.hidden_size,head_params.num_layers,head_params.dropout,head_params.output_size,
  # test_folder
  filter_dict={
    # 'epochs': 1500,
    # 'lr': 0.0001,
    # 'init_network': 'xavier',
    # 'head_params.hidden_size': 128,
    # 'head_params.num_layers': 2,
    # 'head_params.dropout': 0.5,
    'optimizer_fn': 'Adam',
  }
  str_dict = "".join([f"{k[:2]}_{v}_" for k,v in filter_dict.items()])
  # save filter_dict in a config file
  out_path = os.path.join(output_folder, "filtered_"+str_dict)
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  with open(os.path.join(out_path,"_filter_config.json"), 'w') as f:
    json.dump(filter_dict, f, indent=2)
  create_filtered_symlinks(csv_output, filter_dict, out_path)
  
