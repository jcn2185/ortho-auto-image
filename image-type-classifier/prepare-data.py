
import os
import sys
import shutil
from zipfile import ZipFile
from PIL import Image, UnidentifiedImageError
import glob
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map  # or thread_map

RAW_DATA_DIR = "raw-data/"


TYPE_MAP = {
    ".V10": 1,
    ".V12": 2,
    ".V13": 3,
    ".V23": 4,
    ".V24": 5,
    ".V20": 6,
    ".V22": 7,
    ".V21": 8,
    ".V50": 9,
    ".V51": 10
}

broken = []


# Lets loop through the entire raw data directory
# and we'll write our categorized output to the data directory
# we should also adjust target size here

def get_all_files_in_dir(dir_path):
    results = []
    for root, directories, files in os.walk(dir_path, topdown=False):
        for name in files:
            results.append(os.path.join(root, name))
    return results


def load_files_to_ignore():
    results = []
    # open file and read the content in a list
    with open('files-to-ignore.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]

            # add item to the list
            results.append(current_place)

    # now convert list to set
    return set(results)


files_to_ignore = load_files_to_ignore()


def handle_file(filepath, already_in_zip=False):

    # get filetype based on extension
    filetype = os.path.splitext(filepath)[1].upper()

    try:

        if filetype == ".ZIP" and not already_in_zip:
            # Extract it and check out all files that get extracted
            extract_to = 'tmp/' + os.path.basename(filepath)
            with ZipFile(filepath, 'r') as zip:
                zip.extractall(extract_to)

                # Get list of extracted files
                extracted_files = get_all_files_in_dir(extract_to)

                # recursively check all the files now
                for item in extracted_files:
                    handle_file(item, True)

                # Now, clean up the tmp folder you created
                shutil.rmtree(extract_to)
        else:
            # Attempt to map the filetype to a class
            if filetype in TYPE_MAP:
                # is this a valid image?
                img = Image.open(filepath)
                img.verify()

                # Is this just a bad file?
                if os.path.basename(filepath) + ".jpeg" in files_to_ignore:
                    print("Found ignore request for:",
                          os.path.basename(filepath))
                    raise Exception("This is a bad file!",
                                    os.path.basename(filepath) + ".jpeg")

                maptype = str(TYPE_MAP[filetype])

                # everything seems okay, lets try to resize it
                # and move it to our data directory
                new_file_path = "data/" + \
                    str(maptype) + "/" + os.path.basename(filepath) + ".jpeg"

                img = Image.open(filepath)
                # img = img.resize(size=(300, 300))

                img.save(new_file_path, "JPEG")

    except UnidentifiedImageError:
        broken.append(filepath)
    except:
        print("Unexpected error:", sys.exc_info()[0], filepath)
        broken.append(filepath)


if __name__ == '__main__':

    # make sure our output directories exist
    if os.path.exists('data'):
        shutil.rmtree('data')

    # handle temporary folder
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    os.makedirs('tmp')

    for key in TYPE_MAP:
        path = 'data/' + str(TYPE_MAP[key]) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

    # debug
    all_raw_files = get_all_files_in_dir('F:/Raw-Data/')

    # Debug total files found
    print("Found ", len(all_raw_files), "to process")

    # Now parallelize it
    print(os.path.splitext("some-random/test/thing/what.org.png"))

    # # with Pool(8) as p:
    # with Pool(cpu_count() - 1) as p:
    #     p.map(handle_file, all_raw_files)

    r = process_map(handle_file, all_raw_files, max_workers=cpu_count() - 1)
    # r = process_map(
    #     handle_file, ['raw-data/5-14-2021-dolphin/385113.ZIP'], max_workers=1)

    print("discarded files: ", len(broken))

    # now, let's print out our summary statistics
    for idx in range(1, 11):
        files = get_all_files_in_dir('data/' + str(idx) + '/')
        print(str(idx), ":", str(len(files)))
