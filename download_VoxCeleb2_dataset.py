import requests
from bs4 import BeautifulSoup
import re
import os

# Target download folder
download_folder = "/seidenas/users/fvilli/CeleB_V_Text"

# Create folder if it doesn't exist
os.makedirs(download_folder, exist_ok=True)

# Shared Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/1n3dlPfgGvvcVvDri8bLJ9MCBb5X7L_fE"

def extract_tar_files(folder_url):
    folder_id = folder_url.split("/")[-1]
    embed_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"

    response = requests.get(embed_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    tar_files = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if a.text.endswith(".tar") and "id=" in href:
            file_id_match = re.search(r"id=([^&]+)", href)
            if file_id_match:
                file_id = file_id_match.group(1)
                filename = a.text.strip()
                tar_files.append((file_id, filename))

    return tar_files

def generate_gdown_commands(tar_files, target_dir):
    commands = []
    for file_id, filename in tar_files:
        filepath = os.path.join(target_dir, filename)
        cmd = f"gdown --id {file_id} -O \"{filepath}\""
        commands.append(cmd)
    return commands

# Run it
tar_files = extract_tar_files(folder_url)
commands = generate_gdown_commands(tar_files, download_folder)

# Output final commands
print("### COPY AND PASTE BELOW IN UBUNTU TERMINAL ###\n")
print("\n".join(commands))
