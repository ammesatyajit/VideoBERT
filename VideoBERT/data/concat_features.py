import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--root-feature_path', type=str, required=True, help='path to folder containing all the video folders with the features')
parser.add_argument('-s', '--features-save-path', type=str, required=True, help='directory in which to save concatenated features')
args = parser.parse_args()

root_features = args.root_feature_path
save_path = args.features_save_path

features = []
for root, dirs, files in tqdm(os.walk(root_features)):
    for name in files:
        path = os.path.join(root, name)
        features.append(np.load(path))
features_concat = np.concatenate(features)
print("final size:", features_concat.shape)
np.save(save_path, features_concat)
