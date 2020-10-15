import argparse
import numpy as np
import torch
import VideoBERT.data.globals as data_globals
from transformers import BertTokenizer, BertForPreTraining
from VideoBERT.train.modeling_video_bert import VideoBertForPreTraining
from VideoBERT.train.model_utils import create_video_bert_save_dict_from_bert
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main(colab_args=None):
    if colab_args:
        args = colab_args
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args)

    # setup tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(data_globals.bert_model)
    if args.model_name_or_path is None:
        # start from inital model
        print('### LOADING INITIAL MODEL ###')
        model = VideoBertForPreTraining.from_pretrained(
            pretrained_model_name_or_path=None,
            state_dict=create_video_bert_save_dict_from_bert(data_globals.config),
            config=data_globals.config
        )
    else:
        # start from checkpoint
        print('### LOADING MODEL FROM CHECKPOINT:', args.model_name_or_path, '###')
        model = VideoBertForPreTraining.from_pretrained(args.model_name_or_path)

    print('WEIGHTS:')
    print(model.bert.embeddings.word_embeddings.weight)

    centroids = np.load(data_globals.centers_file)
    print('CENTROIDS:')
    print(centroids)

    model.to(args.device)

    avg_loss = 0
    counter = 0
    temp = 1
    predictmode = 'vid-prior'

    import json
    with open(data_globals.val_youcook, 'r') as fd:
        data = json.load(fd)

        for nr, (id, val) in enumerate(data.items(), start=1):
            print('nr:', nr)
            annots = val['annotations']

            for an in annots:
                sent = an['sentence']
                encoded = tokenizer.encode(sent, add_special_tokens=False)
                vsent = np.array(an['video_ids']) + 30522

                if len(vsent) > 0:
                    print("vsent:", vsent)
                    if predictmode == 'vid-prior':
                        input_ids = torch.tensor(np.hstack([
                            np.array([101]),
                            vsent,
                            np.array([102])
                        ]), dtype=torch.int64).unsqueeze(0)

                        token_type_ids = torch.tensor(np.hstack([
                            np.ones(len(vsent) + 2)
                        ]), dtype=torch.int64).unsqueeze(0)

                    elif predictmode == 'joint-prior':
                        input_ids = torch.tensor(np.hstack([
                            np.array([101]),
                            np.array(encoded),
                            np.array([30522 + 20544]),
                            vsent,
                            np.array([102])
                        ]), dtype=torch.int64).unsqueeze(0)

                        token_type_ids = torch.tensor(np.hstack([
                            np.zeros(len(encoded) + 2),
                            np.ones(len(vsent) + 1)
                        ]), dtype=torch.int64).unsqueeze(0)

                    input_ids = input_ids.to(device)
                    token_type_ids = token_type_ids.to(device)
                    attn_mask = (token_type_ids == -1).to(device)

                    if predictmode == 'vid-prior':
                        output, loss = model(
                            video_input_ids=input_ids,
                            video_token_type_ids=token_type_ids,
                            video_attention_mask=attn_mask,
                        )
                    elif predictmode == 'joint-prior':
                        output, loss = model(
                            joint_input_ids=input_ids,
                            joint_token_type_ids=token_type_ids,
                            joint_attention_mask=attn_mask,
                        )

                    counter += 1
                    avg_loss += loss.item()

                    output = torch.softmax(output, dim=1)
                    print(output)

        avg_loss /= counter
        print("Average loss for evaluation set with {}:".format(predictmode), avg_loss)


if __name__ == "__main__":
    main()

