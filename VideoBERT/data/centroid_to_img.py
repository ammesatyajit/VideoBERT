import numpy as np
import torch
from tqdm import tqdm
import os
import json

saved_features = 'saved_features'
saved_imgs = 'saved_imgs'
centroids = np.load('centroids.npy')
centroid_map = {}

feature_paths = {}
feature_list = []

counter = 0
for root, dirs, files in tqdm(os.walk(saved_features)):
    for name in files:
        path = os.path.join(root, name)
        feature_list.append(torch.from_numpy(np.load(path)))
        feature_paths[counter] = path
        counter += 1


def img_path_from_centroid(features, centroid, img_dir):
    min_dist = 2 ** 12 - 1
    vid_id = None
    features_id = None
    features_row = None

    for i in tqdm(range(counter)):
        centroid_dist = torch.linalg.norm(features[i].cuda() - centroid.cuda(), axis=1)
        if torch.min(centroid_dist) < min_dist:
            feature_path = feature_paths[i]
            min_dist = torch.min(centroid_dist)
            vid_id = feature_path[feature_path.index('/') + 1: feature_path.rindex('/')]
            features_id = feature_path[feature_path.rindex('-') + 1: feature_path.rindex('.')]
            features_row = torch.argmin(centroid_dist)

    return os.path.join(img_dir, vid_id, 'img-{}-{:02}.jpg'.format(features_id, features_row))


img_path_from_centroid(feature_list, centroids[0], saved_imgs)
# for i in range(centroids.shape[0]):
#     print("Centroid:", i)
#     centroid_map[i] = img_path_from_centroid(saved_features, centroids[i], saved_imgs)
#
# json.dump(centroid_map, open('centroid_to_img.json', 'wb'), sort_keys=True, indent=4)
