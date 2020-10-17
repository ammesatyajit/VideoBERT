from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import numpy as np
import random
import VideoBERT.data.globals as data_globals

class VideoBertDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
        self.data = np.load(self.data_path, allow_pickle=True).item()
        self.tokenizer = tokenizer
        self.total_items = self.count()
        print('total items:', self.total_items)

        self.items = [None] * self.total_items

        self.setup_data()

    def create_concat_joint_sentence(self, i, max_token_len=-1):
        start_boundary_index = self.items[i][2]
        end_boundary_index = self.items[i][3]

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
                text_sentence += self.items[ti][0]
                video_sentence += self.items[vi][1]
        else:
            text_sentence = self.items[text_index][0]
            video_sentence = self.items[video_index][1]

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
        start_boundary_index = self.items[i][2]
        end_boundary_index = self.items[i][3]

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

        text_sentence = self.items[text_index][0]
        video_sentence = self.items[video_index][1]

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

        start_boundary_index = self.items[i][2]
        end_boundary_index = self.items[i][3]

        label = -1

        first_sentence = None
        second_sentence = None

        if i == end_boundary_index and i == start_boundary_index:
            first_sentence = self.items[i][mode_index]
            second_sentence = self.items[i][mode_index]
            label = 1 # not natural next sentence
        else:
            if i == end_boundary_index:
                i -= 1

            first_sentence = self.items[i][mode_index]

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

            second_sentence = self.items[next_index][mode_index]

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
                np.array(second_sentence),
                np.array(self.tokenizer.sep_token_id)
            ]), dtype=torch.int64
        ), torch.tensor(label, dtype=torch.int64), torch.tensor(np.hstack([
            first_token_type_ids,
            second_token_type_ids
        ]), dtype=torch.int64)


    def setup_data(self):
        i = 0
        for _, val in self.data.items():
            text_items = val['text']
            video_items = val['video']

            start_boundary_index = i
            end_boundary_index = i + len(text_items)-1

            for t, v in zip(text_items, video_items):
                self.items[i] = (t, v, start_boundary_index, end_boundary_index)
                i += 1

    def count(self):
        total = 0
        for _, val in self.data.items():
            total += len(val['text'])  # adjust for last item, has no natural follow-up
        return total

    def __len__(self):
        return self.total_items

    def __getitem__(self, i):
        text_sentence, text_label, text_token_type_ids = self.create_next_sentence_pair(i, 'text', max_token_len=37)
        video_sentence, video_label, video_token_type_ids = self.create_next_sentence_pair(i, 'video', max_token_len=37)
        # joint_sentence, joint_label, joint_token_type_ids = self.create_joint_sentence(i, max_token_len=37)
        joint_sentence, joint_label, joint_token_type_ids = self.create_concat_joint_sentence(i, max_token_len=-1)

        return text_sentence, text_label, text_token_type_ids, video_sentence, video_label, video_token_type_ids, joint_sentence, joint_label, joint_token_type_ids

