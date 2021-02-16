import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--root-features', type=str, required=True,
                    help='path to folder containing all the video folders with the features')
parser.add_argument('-c', '--centroid-file', type=str, required=True,
                    help='the .npy file containing all the centroids')
parser.add_argument('-s', '--save-file', type=str, required=True,
                    help='json file to save the labelled data to')
args = parser.parse_args()

features_root = args.root_features
dirs = os.listdir(features_root)

kmeans = MiniBatchKMeans()
kmeans.cluster_centers_ = np.load(args.centroid_file)

save_path = args.save_file

data_dict = {}

for folder in tqdm(dirs):
    data_dict[folder] = []
    feature_files = sorted(os.listdir(os.path.join(features_root, folder)))
    for features in feature_files:
        data_dict[folder].extend(kmeans.predict(np.load(os.path.join(features_root, folder, features))))
    data_dict[folder] = list(map(lambda x: int(x), data_dict[folder]))

json.dump(data_dict, open(save_path, 'w'), sort_keys=True, indent=4)
