import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--root-feature_path', type=str, required=True, help='path to folder containing all the video folders with the features')
parser.add_argument('-s', '--features-save-path', type=str, required=True, help='directory in which to save concatenated features')
args = parser.parse_args()

root_features = args.root_feature_path
save_path = args.features_save_path

with torch.no_grad():
    # features_concat = None
    # for root, dirs, files in tqdm(os.walk(root_features)):
    #     for name in files:
    #         path = os.path.join(root, name)
    #         features = torch.from_numpy(np.load(path)).cuda()
    #         if features_concat is None:
    #             features_concat = features
    #         else:
    #             features_concat = torch.cat((features_concat, features))

    features_concat = []
    for root, dirs, files in tqdm(os.walk(root_features)):
        for name in files:
            path = os.path.join(root, name)
            features_concat.append(torch.from_numpy(np.load(path)).cuda())

    print("finished reading all features")
    print(features_concat[0].shape)
    features_concat = torch.cat(features_concat)
    print("final size:", features_concat.shape)
    np.save(save_path, features_concat)
