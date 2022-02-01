
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map  # or thread_map
import json
import requests


passwords = open('passwords.json')
coco_login = json.load(passwords)

base_url = "http://157.230.187.183:5000"
api_url = base_url + "/api"


# create session
s = requests.Session()

# make login request
s.post(api_url + '/user/login', json=coco_login)


files = {'file': open(
    '/Users/jcnelson/Documents/JNDocs/Code/ortho-auto-image/image-type-classifier/13496400.V10.jpeg', 'rb')}
values = {'folder': 'View 1'}

r = s.post(api_url + "/images/", files=files)

print(r)

r = s.get(api_url + "/images/")

print(r)
