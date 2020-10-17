import argparse
import numpy as np
import torch
import pandas as pd
import tqdm
import VideoBERT.data.globals as data_globals
from transformers import BertTokenizer
from VideoBERT.train.custom_vid_transformer import VideoTransformer
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
        model = VideoTransformer(
            config=data_globals.config,
            args=args
        )
    else:
        # start from checkpoint
        print('### LOADING MODEL FROM CHECKPOINT:', args.model_name_or_path, '###')
        model = VideoTransformer.from_pretrained(
            config=data_globals.config,
            args=args
        )

    centroids = np.load(data_globals.centers_file)
    print('CENTROIDS:')
    print(centroids)

    model.to(args.device)

    losses = {"total_avg_loss": 0,
              "text_avg_loss": 0,
              "video_avg_loss": 0,
              "joint_avg_loss": 0}
    counter = 0
    temp = 1

    data = pd.read_csv('/content/drive/My Drive/VideoBERT/val_data.csv', delimiter=',')
    data = data.to_dict('records')

    for nr, val in tqdm.tqdm(enumerate(data, start=1)):
        annots = eval(val['annotations'])

        for an in annots:
            sent = an['sentence']
            encoded = tokenizer.encode(sent, add_special_tokens=False)
            vsent = np.array(an['video_ids']) + 30522

            if len(vsent) > 0:
                text_ids = torch.tensor(np.hstack([
                    np.array([101]),
                    np.array(encoded),
                    np.array([102])
                ]), dtype=torch.int64).unsqueeze(0).to(device)
                text_type_ids = torch.zeros_like(text_ids).to(device)
                text_attn_mask = (text_type_ids == -1).to(device)

                vid_ids = torch.tensor(np.hstack([
                    np.array([101]),
                    vsent,
                    np.array([102])
                ]), dtype=torch.int64).unsqueeze(0).to(device)
                vid_type_ids = torch.ones_like(vid_ids).to(device)
                vid_attn_mask = (vid_type_ids == -1).to(device)

                joint_ids = torch.tensor(np.hstack([
                    np.array([101]),
                    np.array(encoded),
                    np.array([30522 + 20544]),
                    vsent,
                    np.array([102])
                ]), dtype=torch.int64).unsqueeze(0).to(device)
                joint_type_ids = torch.tensor(np.hstack([
                    np.zeros(len(encoded) + 2),
                    np.ones(len(vsent) + 1)
                ]), dtype=torch.int64).unsqueeze(0).to(device)
                joint_attn_mask = (joint_type_ids == -1).to(device)

                outputs = model(
                    text_input_ids=text_ids,
                    video_input_ids=vid_ids,
                    joint_input_ids=joint_ids,

                    text_token_type_ids=text_type_ids,
                    video_token_type_ids=vid_type_ids,
                    joint_token_type_ids=joint_type_ids,

                    text_attention_mask=text_attn_mask,
                    video_attention_mask=vid_attn_mask,
                    joint_attention_mask=joint_attn_mask,
                )

                loss = outputs[0]
                text_loss = outputs[2]
                video_loss = outputs[4]
                joint_loss = outputs[6]

                counter += 1
                losses["total_avg_loss"] += loss.item()
                losses["text_avg_loss"] += text_loss.item()
                losses["video_avg_loss"] += video_loss.item()
                losses["joint_avg_loss"] += joint_loss.item()

    losses["total_avg_loss"] /= counter
    losses["text_avg_loss"] /= counter
    losses["video_avg_loss"] /= counter
    losses["joint_avg_loss"] /= counter

    print("Average loss far val set\n--------------------------"
          "\nTotal avg loss: {}"
          "\nText avg loss: {}"
          "\nVideo avg loss: {}"
          "\nJoint avg loss: {}".format(losses["total_avg_loss"], losses["text_avg_loss"], losses["video_avg_loss"], losses["joint_avg_loss"]))


if __name__ == "__main__":
    main()

