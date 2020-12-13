import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.cluster import MiniBatchKMeans


def minibatch_kmeans(root, prefix, k, batch_size, epochs):
    """
    docstring
    """
    paths = []
    for root, dirs, files in tqdm(os.walk(root)):
        for name in files:
            if name.find(prefix) != -1:
                paths.append(os.path.join(root, name))

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    vectors = None

    print("starting kmeans")
    for i in range(epochs):
        print("epoch:", i)
        for path in tqdm(paths):
            if vectors is None:
                vectors = np.load(path)
            else:
                vectors = np.concatenate([vectors, np.load(path)])

            if vectors.shape[0] >= batch_size:
                vectors = vectors[:batch_size, :]
                kmeans.partial_fit(vectors)
                vectors = None
        if vectors is not None and vectors.shape[0] >= k:
            kmeans.partial_fit(vectors)
            vectors = None

    print("labelling data")
    labelled_data = {}
    for path in tqdm(paths):
        labelled_data[path] = list(kmeans.predict(np.load(path)))

    return kmeans.cluster_centers_, labelled_data


def save_sorted_vectors(centroids, labelled_data, batch_size, save_dir, save_prefix):
    k = centroids.shape[0]
    save_path = os.path.join(save_dir, save_prefix) + '-{}-Id:{}'
    for i in range(k):
        sorted_vecs = []
        counter = 1
        for key in tqdm(labelled_data):
            pred_centroids = labelled_data[key]
            vectors = np.load(key)
            for j in range(len(pred_centroids)):
                if pred_centroids[j] == i:
                    sorted_vecs.append(np.expand_dims(vectors[j], axis=0))
                    if len(sorted_vecs) == batch_size:
                        sorted_vecs = np.concatenate(sorted_vecs)
                        np.save(save_path.format(i, counter), sorted_vecs)
                        sorted_vecs = []
                        counter += 1

        if sorted_vecs != []:
            sorted_vecs = np.concatenate(sorted_vecs)
            np.save(save_path.format(i, counter), sorted_vecs)
            sorted_vecs = []


def delete_used_files(root, prefix):
    print("deleting finished files")
    for root, dirs, files in tqdm(os.walk(root)):
        for name in files:
            if name.find(prefix) != -1:
                os.remove(os.path.join(root, name))


def hkmeans(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir):
    counter = 1

    def hkmeans_recursive(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir, cur_h=1):
        nonlocal counter
        print("Current H:", cur_h)
        print(prefix)
        if cur_h != h:
            centroids, labelled_data = minibatch_kmeans(root, prefix, k, batch_size, epochs)
            print("minibatch kmeans done!")
            save_sorted_vectors(centroids, labelled_data, batch_size, save_dir, save_prefix)
            save_prefix += '-{}'
            for i in range(k):
                hkmeans_recursive(save_dir, save_prefix.format(i) + '-', h, k, batch_size, epochs, save_dir,
                                  save_prefix.format(i), centroid_dir, cur_h=cur_h + 1)
                delete_used_files(save_dir, save_prefix.format(i) + '-')
        else:
            centroids, labelled_data = minibatch_kmeans(root, prefix, k, batch_size, epochs)
            print("minibatch kmeans done!")
            np.save(os.path.join(centroid_dir, 'centroids-{}'.format(counter)), centroids)
            counter += 1

    hkmeans_recursive(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root-feature_path', type=str, required=True,
                        help='path to folder containing all the video folders with the features')
    parser.add_argument('-p', '--features-prefix', type=str, required=True,
                        help='prefix that contains the desired files to read')
    parser.add_argument('-b', '--batch-size', type=int, default=500,
                        help='batch_size to use for the minibatch kmeans')
    parser.add_argument('-s', '--save-dir', type=str, required=True,
                        help='save directory for hierarchical kmeans vectors')
    parser.add_argument('-c', '--centroid-dir', type=str, required=True,
                        help='directory to save the centroids in')
    args = parser.parse_args()

    root = args.root_feature_path
    prefix = args.features_prefix
    batch_size = args.batch_size
    save_dir = args.save_dir
    centroid_dir = args.centroid_dir

    hkmeans(root, prefix, 4, 12, batch_size, 15, save_dir, 'vecs', centroid_dir)


if __name__ == "__main__":
    main()
