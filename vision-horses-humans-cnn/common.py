import shutil
import os
import urllib.request
import zipfile


def download_extract_dataset(url, extract_dir, archive_dir='data/archive', force_download=False, force_delete=True):
    os.makedirs(archive_dir, exist_ok=True)

    # Delete extract_dir if force_delete is True
    if force_delete and os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    # Download file if force_download is True or file does not exist
    # Get filename from url
    filename = os.path.join(archive_dir, url.split('/')[-1])

    if force_download or not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    # extract zip file
    if filename.endswith('.zip'):
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(extract_dir)
        zip_ref.close()
    else:
        print('Not a zip file. Skipping %s...'.format(filename))