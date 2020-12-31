import json
import os
from punctuator import Punctuator

punc = Punctuator('model.pcl')
captions = json.load(open('filtered_captions.json', 'r'))
labelled_data = json.load(open('labelled_data.json', 'r'))
vid_ids = os.listdir('saved_imgs')
punctuated_data = {}


def timestamp_to_idx(time):
    return int(0.5 + time / 1.5)


punctuated_data[vid_ids[0]] = []

text = captions[vid_ids[0]]['text']
start = captions[vid_ids[0]]['start']
end = captions[vid_ids[0]]['end']
start.append(end[-2])
end.insert(0, start[1])

punc_text = [sentence.strip() + '.' for sentence in punc.punctuate(' '.join(text)).split('.') if sentence != '']
print(text)
print(start)
print(end)

text_len = [len(phrase.split(' ')) for phrase in text]
text_len = [sum(text_len[:i]) for i in range(len(text_len) + 1)]
punc_len = [len(phrase.split(' ')) for phrase in punc_text]
punc_len = [sum(punc_len[:i]) for i in range(len(punc_len) + 1)]

for i in range(len(punc_text)):
    start_idx = None
    end_idx = None
    for j in range(len(text)):
        if punc_len[i] >= text_len[j]:
            start_idx = j
    for j in range(len(text)):
        if punc_len[i + 1] <= text_len[j + 1]:
            end_idx = j
            break
    print(punc_text[i] + ':', start[start_idx], end[end_idx])
    punctuated_data[vid_ids[0]].append({'sentence': punc_text[i], 'vid_tokens': labelled_data[vid_ids[0]][timestamp_to_idx(start[start_idx]):timestamp_to_idx(end[end_idx])+1]})

print(punctuated_data)
