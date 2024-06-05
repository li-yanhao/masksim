import os
import requests
from tqdm import tqdm
import subprocess

import pandas as pd
from src.thread_pool_plus import ThreadPoolPlus


# download train.zip
# unzip train.zip
# download synthbuster

cache_folder = "cache/"
data_folder = "data/"
train_link = "https://cirrus.universite-paris-saclay.fr/s/2eabgG8fZy8nXME/download/train.zip"
synthbuster_link = "https://zenodo.org/records/10066460/files/synthbuster.zip"
newsynth_csv = "newsynth.csv"

def download_file(url, folder, filename=None, progress=False):
    # If no filename is given, use the last part of the URL as the filename
    if filename is None:
        filename = url.split('/')[-1]

    # Create the full path for the file
    path = os.path.join(folder, filename)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the total file size from headers (if available)
        total_size = int(response.headers.get('content-length', 0))

        if progress:
            # Initialize the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as progress_bar:
                # Open the file with write-binary mode
                with open(path, 'wb') as file:
                    # Iterate over the content in chunks
                    for data in response.iter_content(chunk_size=40960):  # Use a reasonable chunk size
                        # Write data chunk to file
                        file.write(data)
                        # Update the progress bar
                        progress_bar.update(len(data))
            print(f"File downloaded successfully: {path}")
        else:
            with open(path, 'wb') as file:
                # Iterate over the content in chunks
                for data in response.iter_content(chunk_size=40960):  # Use a reasonable chunk size
                    # Write data chunk to file
                    file.write(data)
    else:
        print(f"Failed to download file: HTTP {response.status_code}")


def download_newsynth():
    df = pd.read_csv(newsynth_csv)
    
    column_names = [
        'URL Stable Diffusion 1.*',
        'URL Stable Diffusion 2.1',
        'URL Stable Diffusion XL',
        'URL Dall.e 2',
        'URL Dall.e 3',
        'URL midjourney',
        'URL firefly',
    ]
    class_names = ["sd1", "sd2", "sdxl", "dalle2", "dalle3", "midjourney", "firefly"]

    for class_name, col_name in zip(class_names, column_names):
        url_list = df[col_name]
        save_folder = f"data/newsynth/{class_name}"
        os.makedirs(save_folder, exist_ok=True)
        print(f"Downloading images to {save_folder}")
        pool = ThreadPoolPlus(workers=8)
        for url in url_list:
            url = "http://" + url
            # print(url)
            # download_file(url, save_folder)
            pool.submit(download_file, url, save_folder)
        pool.join()


def download_synthbuster():
    os.makedirs(cache_folder, exist_ok=True)
    # download
    download_file(synthbuster_link, cache_folder)
    # decompress .zip
    os.makedirs("data/", exist_ok=True)
    subprocess.run("unzip cache/synthbuster.zip -d data/", shell=True)


def download_pristine():
    os.makedirs(cache_folder, exist_ok=True)
    download_file(train_link, "cache/")
    
    ## decompress .zip
    ## unzip train.zip -d data/
    os.makedirs("data/", exist_ok=True)
    subprocess.run("unzip cache/train.zip -d data/", shell=True)

if __name__ == "__main__":
    # download_pristine()
    # download_synthbuster()
    download_newsynth()
    pass
