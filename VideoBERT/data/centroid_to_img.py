import numpy as np
import torch
from tqdm import tqdm
import os
import json

saved_features = 'saved_features'
saved_imgs = 'saved_imgs'
centroids = torch.from_numpy(np.load('centroids.npy')).cuda()
centroid_map = {}


def img_path_from_centroid(feature_dir, centroid, img_dir):
    min_dist = 2 ** 12 - 1
    vid_id = None
    features_id = None
    features_row = None

    for root, dirs, files in tqdm(os.walk(feature_dir)):
        for name in files:
            path = os.path.join(root, name)
            features = torch.from_numpy(np.load(path)).cuda()
            centroid_dist = torch.linalg.norm(features - centroid, axis=1).cuda()
            if torch.min(centroid_dist) < min_dist:
                min_dist = torch.min(centroid_dist)
                vid_id = path[path.index('/') + 1: path.rindex('/')]
                features_id = path[path.rindex('-') + 1: path.rindex('.')]
                features_row = torch.argmin(centroid_dist)

    return os.path.join(img_dir, vid_id, 'img-{}-{:02}.jpg'.format(features_id, features_row))


for i in range(centroids.shape[0]):
    print("Centroid:", i)
    centroid_map[i] = img_path_from_centroid(saved_features, centroids[i], saved_imgs)

json.dump(centroid_map, open('centroid_to_img.json', 'wb'), sort_keys=True, indent=4)
