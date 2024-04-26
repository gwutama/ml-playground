import shutil
import os
import urllib.request
import zipfile
import random


def download_unzip(url, extract_dir, archive_dir='data/archive', force_download=False, force_delete=False):
    filename = download(url, archive_dir, force_download, force_delete)
    unzip(filename, extract_dir, force_delete)


def download(url, archive_dir='data/archive', force_download=False, force_delete=False):
    os.makedirs(archive_dir, exist_ok=True)

    # Get filename from url
    filename = os.path.join(archive_dir, url.split('/')[-1])

    # Delete file if force_delete is True
    if force_delete and os.path.isfile(filename):
        os.remove(filename)

    # Download file if force_download is True or file does not exist
    if force_download or not os.path.isfile(filename):
        print('Downloading {}...'.format(filename))
        urllib.request.urlretrieve(url, filename)

    return filename


def unzip(filename, extract_dir, force_delete=False):
    # Delete extract_dir if force_delete is True
    if force_delete and os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    # check whether file exists, otherwise throw error
    if not os.path.isfile(filename):
        raise FileNotFoundError('File {} does not exist'.format(filename))

    # extract zip file
    print('Extracting {} to {}...'.format(filename, extract_dir))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(extract_dir)
    zip_ref.close()


def random_split_dataset(source_dir, dest_training_dir, dest_validation_dir, split_size=0.9):
    files = []
    for filename in os.listdir(source_dir):
        file = os.path.join(source_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = os.path.join(source_dir, filename)
        destination = os.path.join(dest_training_dir, filename)
        shutil.copyfile(this_file, destination)

    for filename in testing_set:
        this_file = os.path.join(source_dir, filename)
        destination = os.path.join(dest_validation_dir, filename)
        shutil.copyfile(this_file, destination)


def recreate_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)
