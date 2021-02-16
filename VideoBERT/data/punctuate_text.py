import json
import os
from tqdm import tqdm
from punctuator import Punctuator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--captions-path', type=str, required=True,
                    help='path to filtered captions')
parser.add_argument('-p', '--punctuator-model', type=str, required=True,
                    help='path to punctuator .pcl model')
parser.add_argument('-l', '--labelled-data', type=str, required=True,
                    help='path to labelled data json file')
parser.add_argument('-f', '--root-features', type=str, required=True,
                    help='directory with all the video features')
parser.add_argument('-s', '--save-path', type=str, required=True,
                    help='json file to save training data to')
args = parser.parse_args()

captions_path = args.captions_path
save_path = args.save_path

punc = Punctuator(args.punctuator_model)
captions = json.load(open(captions_path, 'r'))
labelled_data = json.load(open(args.labelled_data, 'r'))
vid_ids = os.listdir(args.root_features)

start = 0
if os.path.exists(save_path):
    train_data = json.load(open(save_path))
    print('starting from vid id', len(train_data))
    start = len(train_data)
else:
    train_data = {}


def timestamp_to_idx(time):
    return int(0.5 + time / 1.5)


def punc_text_and_timestamp(text, start, end, vid_idx):
    punc_text = [sentence.strip() + '.' for sentence in punc.punctuate(' '.join(text)).split('.') if sentence != '']

    text_len = [len(phrase.split(' ')) for phrase in text]
    text_len = [sum(text_len[:i]) for i in range(len(text_len) + 1)]
    punc_len = [len(phrase.split(' ')) for phrase in punc_text]
    punc_len = [sum(punc_len[:i]) for i in range(len(punc_len) + 1)]

    out = []

    for i in range(len(punc_text)):
        start_idx = 0
        end_idx = -1
        for j in range(len(text)):
            if punc_len[i] >= text_len[j]:
                start_idx = j
        for j in range(len(text)):
            if punc_len[i + 1] <= text_len[j + 1]:
                end_idx = j
                break

        out.append({'sentence': punc_text[i],
                    'vid_tokens': labelled_data[vid_idx][timestamp_to_idx(start[start_idx]):
                                                         timestamp_to_idx(end[end_idx])]})

    return out


for i in tqdm(range(start, len(vid_ids))):
    vid_id = vid_ids[i]
    try:
        raw_text = captions[vid_id]['text']
        start_list = captions[vid_id]['start']
        end_list = captions[vid_id]['end']
        start_list.append(end_list[-2])
        end_list.insert(0, start_list[1])

        train_data[vid_id] = punc_text_and_timestamp(raw_text, start_list, end_list, vid_id)
    except KeyboardInterrupt:
        print('saving already grouped training data')
        break


json.dump(train_data, open(save_path, 'w'))
