import simplejson as json
import os
import ijson
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--captions-path', type=str, required=True,
                    help='path to unfiltered captions')
parser.add_argument('-s', '--save-path', type=str, required=True,
                    help='path to save filtered captions')
args = parser.parse_args()

captions_path = args.captions_path
save_path = args.save_path

ids = os.listdir('saved_features')
filtered_captions = {}

with open(captions_path, 'r') as input_file:
    captions_json = ijson.kvitems(input_file, '')
    try:
        for vid_id, captions in tqdm(captions_json):
            if vid_id in ids:
                filtered_captions[vid_id] = captions
    except ijson.common.IncompleteJSONError:
        print('nan detected, moving on')

json.dump(filtered_captions, open(save_path, 'w'))
