import numpy as np
import torch
from tqdm import tqdm
import os
import json

saved_features = 'saved_features'
saved_imgs = 'saved_imgs'
centroids = np.load('centroids.npy')
save_path = 'centroid_to_img.json'
centroid_map = {}

feature_paths = {}
feature_list = []

counter = 0


def img_path_from_centroid(features, centroid, img_dir):
    min_dist = 2 ** 12 - 1
    vid_id = None
    features_id = None
    features_row = None

    for i in range(counter):
        centroid_dist = np.linalg.norm(features[i] - centroid, axis=1)
        centroid_min_dist = np.min(centroid_dist)
        if centroid_min_dist < min_dist:
            path = feature_paths[i]
            min_dist = centroid_min_dist
            vid_id = path[path.index('/') + 1: path.rindex('/')]
            features_id = path[path.rindex('-') + 1: path.rindex('.')]
            features_row = np.argmin(centroid_dist)

    return os.path.join(img_dir, vid_id, 'img-{}-{:02}.jpg'.format(features_id, features_row))


for root, dirs, files in tqdm(os.walk(saved_features)):
    for name in files:
        path = os.path.join(root, name)
        feature_list.append(np.load(path))
        feature_paths[counter] = path
        counter += 1

start_id = 0
try:
    centroid_map = json.load(open(save_path, 'r'))
    start_id = len(centroid_map)
    print('starting at centroid', start_id)
except:
    print('starting with empty centroid_map')

for i in tqdm(range(start_id, centroids.shape[0])):
    try:
        centroid_map[str(i)] = img_path_from_centroid(feature_list, centroids[i], saved_imgs)
    except KeyboardInterrupt:
        break

print('saving centroids and corresponding images')
json.dump(centroid_map, open(save_path, 'w'), sort_keys=True, indent=4)
