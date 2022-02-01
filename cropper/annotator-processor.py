# Written by John Nelson (jcnelson2185@gmail.com)


from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map  # or thread_map
import json
import requests

f = open('annotator.json')
passwords = open('passwords.json')

data = json.load(f)

coco_login = json.load(passwords)

# create a session
s = requests.Session()

base_url = "http://157.230.187.183:5000"
api_url = base_url + "/api"

# Make login request
s.post(api_url + '/user/login', json=coco_login)

# Build a list of the data sets we want to track
known_categories = list(range(29, 35))

# Which datasets should we track
to_track = ['1-200', '200-400', '400-600', '600-800', '800-1000']

# directory to store images
image_output_dir = "data/right-extra-oral"


def get_coco_datasets(coco_file_path):
    # Now, let's request the different collections
    r = s.get(api_url + '/dataset')
    dataset_name_to_id_mapping = {}
    for i in r.json():
        if i['name'] in to_track:
            # download coco data
            url_request = api_url + '/dataset/' + str(i['id']) + '/coco'
            print(url_request)
            d = s.get(url_request)
            dataset_name_to_id_mapping[i['name']] = d.json()

    # save json data to file
    with open(coco_file_path, 'w') as o:
        json.dump(dataset_name_to_id_mapping, o, indent=4)


def reformat_coco_datasets(coco_file_path, formatted_file_path):
    with open(coco_file_path) as json_file:
        datasets = json.load(json_file)

        image_list = []

        # Loop through each dataset
        for set in datasets:
            print(set)
            current = datasets[set]

            # loop through all of the images
            for image in current['images']:

                # Do we have an annotation for this image?
                found = [x for x in current['annotations']
                         if x['image_id'] == image['id']]
                if len(found) == 6:
                    # Make sure we get all of the right keypoints
                    annotations = []
                    for category in known_categories:

                        found_category = next(
                            (x for x in found if x['category_id'] == category), None)
                        if found_category is not None:
                            annotations.append(
                                {"x": found_category['keypoints'][0] / (image['width'] + 0.0), "y": found_category['keypoints'][1] / (image['height'] + 0.0)})

                    # Make sure annotations are the correct length
                    if len(annotations) != 6:
                        continue
                    # build storage
                    image_list.append({
                        "image_id": image['id'],
                        "path": image['path'],
                        "width": image['width'],
                        "height": image['height'],
                        "filename": image['file_name'],
                        "annotations": annotations,
                        "system_path": image_output_dir + "/" + image['file_name']
                    })
        # save json data to file
        with open(formatted_file_path, 'w') as o:
            json.dump(image_list, o, indent=4)


def download_individual_file(file):
    print("Download image id:", str(file['image_id']))
    r = s.get(api_url + "/image/" + str(file['image_id']))
    with open(image_output_dir + "/" + file['filename'], 'wb') as f:
        f.write(r.content)


def download_coco_files(formated_coco_path):
    with open(formated_coco_path) as json_file:
        dataset = json.load(json_file)

     # dataset should be a list, so we should be able to just process map each file
    r = process_map(download_individual_file, dataset,
                    max_workers=cpu_count() - 1)


if __name__ == "__main__":
    coco_data_path = "data.json"
    formatted_coco = "formatted-coco.json"
    # Download all data
    get_coco_datasets(coco_data_path)

    # Reformat coco datasets
    # reformat_coco_datasets(coco_data_path, formatted_coco)

    # Now, we should load a list of files
    # download_coco_files(formatted_coco)
