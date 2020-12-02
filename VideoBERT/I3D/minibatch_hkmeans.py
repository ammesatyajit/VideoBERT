import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.cluster import MiniBatchKMeans


def minibatch_kmeans(root, prefix, k, batch_size, epochs):
    """
    concatenates all vectors starting with prefix and mini-batch kmeans on
    them with k as input k and the batch size as batch_size. Returns the
    centroids and the labelled data which is a vector of the closest centroid
    to each data point
    """
    paths = []
    for root, dirs, files in tqdm(os.walk(root)):
        for name in files:
            if name.find(prefix) != -1:
                paths.append(os.path.join(root, name))

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    vectors = None
    print("starting kmeans")
    for i in tqdm(range(epochs)):
        for path in paths:
            if vectors is None:
                vectors = np.load(path)
            else:
                vectors = np.concatenate([vectors, np.load(path)])

            if vectors.shape[0] >= batch_size:
                vectors = vectors[:batch_size, :]
                kmeans.partial_fit(vectors)
                vectors = None
    print("labelling data")
    labelled_data = {}
    for path in tqdm(paths):
        labelled_data[path] = list(kmeans.predict(np.load(path)))

    return kmeans.cluster_centers_, labelled_data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root-feature_path', type=str, required=True,
                        help='path to folder containing all the video folders with the features')
    parser.add_argument('-p', '--features-prefix', type=str, required=True,
                        help='prefix that contains the desired files to read')
    args = parser.parse_args()

    root = args.root_feature_path
    prefix = args.features_prefix

    centroids, labelled_data = minibatch_kmeans(root, prefix, 12, 20000, 10)
    print(centroids.shape)
    print(labelled_data.items[0])

if __name__ == "__main__":
    main()