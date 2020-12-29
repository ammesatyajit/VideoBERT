import json
import os
import ijson

ids = os.listdir('saved_features')
filtered_captions = {}

with open('raw_caption_superclean.json', 'r') as input_file:
    captions_json = ijson.kvitems(input_file, '')
    for vid_id, captions in captions_json:
        if vid_id in ids:
            filtered_captions[vid_id] = captions

json.dump(filtered_captions, open('filtered_captions.json', 'w'))
