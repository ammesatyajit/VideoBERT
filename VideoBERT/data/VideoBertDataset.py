import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import VideoBERT.data.globals as data_globals


class VideoBertDataset(Dataset):
    def __init__(self, tokenizer, data_path, build_tokenizer=True):
        self.data_path = data_path
        self.data = json.load(open(self.data_path, 'r'))
        self.tokenizer = tokenizer
        self.setup_data()
        if build_tokenizer:
            self.setup_tokenizer()

    def setup_data(self):
        self.tokenizer.sep_token = '<sep>'
        examples = []
        for _, values in self.data.items():
            examples.extend(values)
        self.data = examples

    def setup_tokenizer(self):
        vocab_data = [self.tokenizer.sep_token]
        for example in self.data:
            vocab_data.append(self.tokenizer.tokenize(example['sentence']))
        self.tokenizer.build_vocab(vocab_data)

    def create_text_example(self, i):
        sentence = self.data[i]['sentence']
        sentence = [self.tokenizer.vocab.stoi[token] for token in self.tokenizer.tokenize(sentence)]
        sentence.insert(0, self.tokenizer.vocab.stoi[self.tokenizer.init_token])
        sentence.append(self.tokenizer.vocab.stoi[self.tokenizer.eos_token])

        text_ids = torch.LongTensor(sentence)
        text_tok_type_ids = torch.zeros_like(text_ids)
        text_attn_mask = text_ids == self.tokenizer.vocab.stoi[self.tokenizer.pad_token]

        return text_ids, text_tok_type_ids, text_attn_mask

    def create_video_example(self, i):
        vid_ids = self.data[i]['vid_tokens']
        vid_ids = [vid_id + len(self.tokenizer.vocab) for vid_id in vid_ids]
        vid_ids.insert(0, self.tokenizer.vocab.stoi[self.tokenizer.init_token])
        vid_ids.append(self.tokenizer.vocab.stoi[self.tokenizer.eos_token])

        vid_ids = torch.LongTensor(vid_ids)
        vid_tok_type_ids = torch.ones_like(vid_ids)
        vid_attn_mask = vid_ids == self.tokenizer.vocab.stoi[self.tokenizer.pad_token]

        return vid_ids, vid_tok_type_ids, vid_attn_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text_ids, text_tok_type_ids, text_attn_mask = self.create_text_example(item)
        vid_ids, vid_tok_type_ids, vid_attn_mask = self.create_video_example(item)

        joint_ids = torch.hstack([text_ids[:-1],
                                  torch.LongTensor([self.tokenizer.vocab.stoi[self.tokenizer.sep_token]]),
                                  vid_ids[1:]])

        joint_tok_type_ids = torch.hstack([text_tok_type_ids,
                                           vid_tok_type_ids[1:]])

        joint_attn_mask = joint_ids == self.tokenizer.vocab.stoi[self.tokenizer.pad_token]

        return text_ids, \
               text_tok_type_ids, \
               text_attn_mask, \
               vid_ids, \
               vid_tok_type_ids, \
               vid_attn_mask, \
               joint_ids, \
               joint_tok_type_ids, \
               joint_attn_mask


class VideoBertDatasetOld(Dataset):
    def __init__(self, tokenizer, data_path):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
        self.data = pd.read_csv(self.data_path, delimiter=',')
        self.tokenizer = tokenizer
        self.setup_data()

    def create_concat_joint_sentence(self, i, max_token_len=-1):
        start_boundary_index = self.data[i][2]
        end_boundary_index = self.data[i][3]

        text_index = -1
        video_index = -1
        label = -1
        edge_case = False

        if i == end_boundary_index:
            if i == start_boundary_index:
                text_index = i
                video_index = i
                label = 0
                edge_case = True

        if not edge_case:
            # choose between to concat or not
            r = random.uniform(0, 1)

            if r > 0.5:
                # no concat

                # choose between temp aligned or not
                r = random.uniform(0,1)
                if r > 0.5:
                    text_index = i
                    video_index = i
                    label = 0  # temporarily aligned
                else:
                    indices = [k for k in range(start_boundary_index, end_boundary_index + 1) if k != i]
                    text_index = i
                    video_index = random.choice(indices)
                    label = 1  # not temporarily aligned
            else:
                # do concat

                # choose between temp aligned or not
                r = random.uniform(0,1)

                if r > 0.5:
                    # temp aligned
                    start_index = -1
                    concat_index = -1

                    if i == end_boundary_index:
                        concat_index = i
                        start_index = i-1
                    else:
                        r = random.uniform(0,1)
                        if r > 0.5:
                            start_index = i
                            concat_index = i+1
                        else:
                            start_index = i-1
                            concat_index = i

                    text_index = [start_index, concat_index]
                    video_index = [start_index, concat_index]
                    label = 0 # temp aligned
                else:
                    indices = [k for k in range(start_boundary_index, end_boundary_index + 1) if k != i and k != i+1]
                    try:
                        concat_index = random.choice(indices)
                    except:
                        concat_index = 0
                        print('bad')
                    text_index = [i, concat_index]
                    video_index = [i, concat_index]
                    label = 1 # not temp aligned

        text_sentence = None
        video_sentence = None

        if type(text_index) == list and type(video_index) == list:
            text_sentence = []
            video_sentence = []

            for ti, vi in zip(text_index, video_index):
                text_sentence += self.data[ti][0]
                video_sentence += self.data[vi][1]
        else:
            text_sentence = self.data[text_index][0]
            video_sentence = self.data[video_index][1]

        if max_token_len > 0:
            if (len(text_sentence) + len(video_sentence)) > max_token_len:
                # print('truncating JOINT...')
                num_tokens_to_remove = (len(text_sentence) + len(video_sentence)) - max_token_len
                first, second, _ = self.tokenizer.truncate_sequences(ids=text_sentence, pair_ids=video_sentence,
                                                                     num_tokens_to_remove=num_tokens_to_remove)
                text_sentence = first
                video_sentence = second

        text_token_type_ids = np.zeros(len(text_sentence) + 2)
        video_token_type_ids = np.ones(len(video_sentence) + 1)

        return torch.tensor(np.hstack([
            np.array(self.tokenizer.cls_token_id),
            np.array(text_sentence),
            np.array(data_globals.vis_lin_glue_token_id),  # glue embedding id
            np.array(video_sentence),
            np.array(self.tokenizer.sep_token_id)
        ]), dtype=torch.int64), torch.tensor(label, dtype=torch.int64), torch.tensor(np.hstack([
            text_token_type_ids,
            video_token_type_ids,
        ]), dtype=torch.int64)

    def create_joint_sentence(self, i, max_token_len=-1):
        start_boundary_index = self.data[i][2]
        end_boundary_index = self.data[i][3]

        text_index = -1
        video_index = -1
        label = -1
        edge_case = False

        if i == end_boundary_index:
            if i == start_boundary_index:
                text_index = i
                video_index = i
                label = 0
                edge_case = True
            else:
                i -= 1

        if not edge_case:
            r = random.uniform(0,1)
            if r > 0.5:
                text_index = i
                video_index = i
                label = 0 # temporarily aligned
            else:
                text_index = i
                indices = [k for k in range(start_boundary_index, end_boundary_index + 1) if k != i]
                video_index = random.choice(indices)
                label = 1 # not temporarily aligned

        text_sentence = self.data[text_index][0]
        video_sentence = self.data[video_index][1]

        if max_token_len > 0:
            if (len(text_sentence) + len(video_sentence)) > max_token_len:
               num_tokens_to_remove = (len(text_sentence) + len(video_sentence)) - max_token_len
               first, second, _ = self.tokenizer.truncate_sequences(ids=text_sentence, pair_ids=video_sentence, num_tokens_to_remove=num_tokens_to_remove)
               text_sentence = first
               video_sentence = second

        text_token_type_ids = np.zeros(len(text_sentence) + 2)
        video_token_type_ids = np.ones(len(video_sentence) + 1)

        return torch.tensor(np.hstack([
            np.array(self.tokenizer.cls_token_id),
            np.array(text_sentence),
            np.array(data_globals.vis_lin_glue_token_id),  # glue embedding id
            np.array(video_sentence),
            np.array(self.tokenizer.sep_token_id)
        ]), dtype=torch.int64), torch.tensor(label, dtype=torch.int64), torch.tensor(np.hstack([
            text_token_type_ids,
            video_token_type_ids,
        ]), dtype=torch.int64)

    def create_next_sentence_pair(self, i, mode, max_token_len=-1):
        mode_index = -1

        if mode == 'text':
            mode_index = 0
        elif mode == 'video':
            mode_index = 1

        start_boundary_index = self.data[i][2]
        end_boundary_index = self.data[i][3]

        label = -1

        first_sentence = None
        second_sentence = None

        if i == end_boundary_index and i == start_boundary_index:
            first_sentence = self.data[i][mode_index]
            second_sentence = self.data[i][mode_index]
            label = 1 # not natural next sentence
        else:
            if i == end_boundary_index:
                i -= 1

            first_sentence = self.data[i][mode_index]

            r = random.uniform(0, 1)

            if r > 0.5:
                # take natural next sentence
                next_index = i+1
                label = 0
            else:
                # take random sentence
                indices = [k for k in range(start_boundary_index, end_boundary_index+1) if k != i+1]
                next_index = random.choice(indices)
                label = 1

            second_sentence = self.data[next_index][mode_index]

        if max_token_len > 0:
            if (len(first_sentence) + len(second_sentence)) > max_token_len:
               #print('truncating SINGLE...')
               num_tokens_to_remove = (len(first_sentence) + len(second_sentence)) - max_token_len
               first, second, _ = self.tokenizer.truncate_sequences(ids=first_sentence, pair_ids=second_sentence, num_tokens_to_remove=num_tokens_to_remove)
               first_sentence = first
               second_sentence = second

        first_token_type_ids = np.zeros(len(first_sentence) + 2)
        second_token_type_ids = np.ones(len(second_sentence) + 1)

        return torch.tensor(
            np.hstack([
                np.array(self.tokenizer.cls_token_id),
                np.array(first_sentence),
                np.array(self.tokenizer.sep_token_id),
                np.array(second_sentence),
                np.array(self.tokenizer.sep_token_id)
            ]), dtype=torch.int64
        ), torch.tensor(label, dtype=torch.int64), torch.tensor(np.hstack([
            first_token_type_ids,
            second_token_type_ids
        ]), dtype=torch.int64)

    def setup_data(self):
        self.data = [list(x) for x in self.data.values]
        for data_tuple in self.data:
            data_tuple[0] = eval(data_tuple[0])
            data_tuple[1] = eval(data_tuple[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text_sentence, text_label, text_token_type_ids = self.create_next_sentence_pair(i, 'text', max_token_len=37)
        video_sentence, video_label, video_token_type_ids = self.create_next_sentence_pair(i, 'video', max_token_len=37)
        # joint_sentence, joint_label, joint_token_type_ids = self.create_joint_sentence(i, max_token_len=37)
        joint_sentence, joint_label, joint_token_type_ids = self.create_concat_joint_sentence(i, max_token_len=-1)

        return text_sentence, text_label, text_token_type_ids, video_sentence, video_label, video_token_type_ids, joint_sentence, joint_label, joint_token_type_ids

