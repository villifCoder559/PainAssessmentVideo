import os
import subprocess
from tqdm import tqdm
import argparse

def run_command(command):
  result = subprocess.run(command, shell=True)
  if result.returncode != 0:
    print(f"\nWarning: Command failed: {command}")
  return result

def download_and_extract(root_folder_path):
  os.makedirs(root_folder_path, exist_ok=True) 

  link_template = "https://huggingface.co/datasets/issai/Speaking_Faces/resolve/main/image_only/sub_100_io.zip"
  start, end = 1, 142

  for idx in tqdm(range(start, end + 1), desc="Processing subjects"):
    zip_name = f"sub_{idx}_io.zip"
    folder_name = f"sub_{idx}_io"
    zip_path = os.path.join(root_folder_path, zip_name)
    folder_path = os.path.join(root_folder_path, folder_name)
    link = link_template.replace("sub_100_io", f"sub_{idx}_io")

    # Download if needed
    if not os.path.exists(zip_path) and not os.path.exists(folder_path):
      print(f"Downloading {zip_name}...")
      run_command(f"wget {link} -O {zip_path}")

    # Unzip only specific folders into correct destination
    if not os.path.exists(folder_path) and os.path.exists(zip_path):
      # os.makedirs(folder_path, exist_ok=True)
      folder_path = os.path.dirname(folder_path)
      print(f"Extracting {zip_name} to {folder_path}...")
      for trial in [1, 2]:
        print(f"Processing trial {trial} for subject {idx}...")
        zip_sub_path = f"sub_{idx}_io/trial_{trial}/rgb_image_aligned/*"
        # target_sub_path = f"trial_{trial}/rgb_image_aligned"
        # os.makedirs(os.path.join(folder_path, target_sub_path), exist_ok=True)
        unzip_cmd = f'unzip -q "{zip_path}" "{zip_sub_path}" -d "{folder_path}"'
        run_command(unzip_cmd)

      try:
        print(f"Deleting zip file: {zip_path}")
        os.remove(zip_path)
      except Exception as e:
        print(f"Could not delete zip file: {zip_path} — {e}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download and extract selected folders from Speaking Faces dataset.")
  parser.add_argument("--root_folder", type=str, required=True, help="Folder to save and extract data into")

  args = parser.parse_args()
  download_and_extract(args.root_folder)
