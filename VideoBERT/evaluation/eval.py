import argparse
import logging
import os
import random

import numpy as np
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from VideoBERT.data.VideoBertDataset import VideoBertDataset
from VideoBERT.train.custom_vid_transformer import VideoTransformer
from VideoBERT.train.model_utils import *

spacy_en = spacy.load('en')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, eval_dataset: VideoBertDataset):
    """ Evaluate the model """

    # Calculates the batch size for training given number of gpus and batch size for gpus

    pad_id = eval_dataset.tokenizer.vocab.stoi[eval_dataset.tokenizer.pad_token]

    def collate(examples):
        text_examples = [None] * len(examples)
        text_type_ids = [None] * len(examples)

        video_examples = [None] * len(examples)
        video_type_ids = [None] * len(examples)

        joint_examples = [None] * len(examples)
        joint_type_ids = [None] * len(examples)

        for i, (t_ids, t_type_ids, _, v_ids, v_type_ids, _, j_ids, j_type_ids, _) in enumerate(examples):
            text_examples[i] = t_ids
            text_type_ids[i] = t_type_ids

            video_examples[i] = v_ids
            video_type_ids[i] = v_type_ids

            joint_examples[i] = j_ids
            joint_type_ids[i] = j_type_ids

        padded_text_ids = pad_sequence(text_examples, batch_first=True, padding_value=pad_id)
        padded_text_type_ids = pad_sequence(text_type_ids, batch_first=True, padding_value=0)
        padded_text_attn_mask = padded_text_ids == pad_id

        padded_video_ids = pad_sequence(video_examples, batch_first=True, padding_value=pad_id)
        padded_video_type_ids = pad_sequence(video_type_ids, batch_first=True, padding_value=1)
        padded_video_attn_mask = padded_video_ids == pad_id

        padded_joint_ids = pad_sequence(joint_examples, batch_first=True, padding_value=pad_id)
        padded_joint_type_ids = pad_sequence(joint_type_ids, batch_first=True, padding_value=1)
        padded_joint_attn_mask = padded_joint_ids == pad_id

        return padded_text_ids.to(args.device), \
               padded_text_type_ids.to(args.device), \
               padded_text_attn_mask.to(args.device), \
               padded_video_ids.to(args.device), \
               padded_video_type_ids.to(args.device), \
               padded_video_attn_mask.to(args.device), \
               padded_joint_ids.to(args.device), \
               padded_joint_type_ids.to(args.device), \
               padded_joint_attn_mask.to(args.device)

    # initializes dataloader
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_train_batch_size, collate_fn=collate
    )

    global_step = 0

    tr_loss, tr_text_loss, tr_vid_loss, tr_joint_loss = 4 * [0.0]
    set_seed(args)  # Added here for reproducibility
    model.eval()
    eval_iterator = tqdm(eval_dataloader, desc="Iteration")

    for step, \
        [text_ids,
         text_type_ids,
         text_attn_mask,
         video_ids,
         video_type_ids,
         video_attn_mask,
         joint_ids,
         joint_type_ids,
         joint_attn_mask] in enumerate(eval_iterator):

        torch.cuda.empty_cache()

        if text_ids.shape[1] >= 300:
            continue

        outputs = model(
            text_input_ids=text_ids,
            text_token_type_ids=text_type_ids,
            text_attention_mask=text_attn_mask,

            video_input_ids=video_ids,
            video_token_type_ids=video_type_ids,
            video_attention_mask=video_attn_mask,

            joint_input_ids=joint_ids,
            joint_token_type_ids=joint_type_ids,
            joint_attention_mask=joint_attn_mask
        )

        tr_loss += outputs[0]
        tr_text_loss += outputs[2]
        tr_vid_loss += outputs[4]
        tr_joint_loss += outputs[6]

        outputs = None

    return tr_loss / global_step, tr_text_loss / global_step, tr_vid_loss / global_step, tr_joint_loss / global_step


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
        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument(
            "--eval_data_path",
            default=None,
            type=str,
            help="The json file for training the model"
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 1

    set_seed(args)

    # setup tokenizer and model
    tokenizer = torch.load(os.path.join(args.output_dir, "tokenizer.pt"))

    # start from checkpoint
    print('### LOADING MODEL FROM CHECKPOINT:', args.model_name_or_path, '###')
    model = VideoTransformer.from_pretrained(config=data_globals.config, args=args)

    model.to(args.device)

    eval_dataset = VideoBertDataset(tokenizer, build_tokenizer=False, data_path=args.eval_data_path)

    total_avg_loss, text_avg_loss, video_avg_loss, joint_avg_loss = evaluate(args, model, eval_dataset)

    print("\nAverage loss for validation set\n----------------------------------"
          "\nTotal avg loss: {}"
          "\nText avg loss: {}"
          "\nVideo avg loss: {}"
          "\nJoint avg loss: {}".format(total_avg_loss, text_avg_loss, video_avg_loss, joint_avg_loss))


if __name__ == "__main__":
    main()
