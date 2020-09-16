import VideoBERT.I3D.extract_features as i3d
import os.path
import argparse
import pathlib


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file-list-path', type=str, required=True, help='path to file containing video file names')
parser.add_argument('-r', '--root-video-path', type=str, required=True, help='root directory containing video files')
parser.add_argument('-s', '--save-path', type=str, required=True, help='directory in which to save features')
args = parser.parse_args()

video_file_list_path = args.file_list_path
video_root_path = args.root_video_path
features_save_path = args.save_path

pathlib.Path(features_save_path).mkdir(parents=True, exist_ok=True)

with open(video_file_list_path, 'r') as fd:
    video_files = list(map(lambda l: l.strip(), fd.readlines()))

video_paths = [os.path.join(video_root_path, f) for f in video_files]

print(len(video_paths), 'video paths found.')

from_index = 0

for i, path in enumerate(video_paths[from_index:]):
    try:
        print('processing:', path, '[{}/{}]'.format(i+1, len(video_paths[from_index:])))

        folder_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(features_save_path, folder_name)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

        # i3d.features_save_dir = save_path
        i3d.extract_features(path, save_path)

        print('completion status:', path, '[SUCCESS]')
    except Exception as e:
        print(e)
        print('completion status:', path, '[FAILED]')
