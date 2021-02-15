from hkmeans_minibatch.hkmeans import hkmeans
import argparse


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
