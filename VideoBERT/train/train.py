from VideoBERT.evaluation.eval import evaluate
from VideoBERT.train.custom_vid_transformer import VideoTransformer
from VideoBERT.data.VideoBertDataset import VideoBertDataset
from VideoBERT.train.model_utils import *

import argparse
import glob
import logging
import os
import random
import re
import shutil
from typing import List, Tuple

import numpy as np
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from torchtext.data import Field
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

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


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, model, train_dataset: VideoBertDataset, eval_dataset: VideoBertDataset) -> Tuple[int, float]:
    """ Train the model """
    # will graph summary of training and eval at the end of each epoch
    tb_writer = SummaryWriter()

    # Calculates the batch size for training given number of gpus and batch size for gpus
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    pad_id = train_dataset.tokenizer.stoi[train_dataset.tokenizer.pad_token]

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
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    # calculates number of epochs based on number of steps to take in training
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    model.train()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, \
            text_ids, \
            text_type_ids, \
            text_attn_mask, \
            video_ids, \
            video_type_ids, \
            video_attn_mask, \
            joint_ids, \
            joint_type_ids, \
            joint_attn_mask in enumerate(epoch_iterator):

            torch.cuda.empty_cache()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

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

            total_loss = outputs[0]
            text_loss = outputs[2]
            video_loss = outputs[4]
            joint_loss = outputs[6]

            if args.n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
                # text_loss = text_loss / args.gradient_accumulation_steps
                # video_loss = video_loss / args.gradient_accumulation_steps
                # joint_loss = joint_loss / args.gradient_accumulation_steps

            total_loss.backward()

            tr_loss += total_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                print('loss:', total_loss.item(),
                      'text loss:', text_loss.item(),
                      'video loss:', video_loss.item(),
                      'joint loss:', joint_loss.item())

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print('writing tf logs...')
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    total_eval_loss, text_eval_loss, video_eval_loss, joint_eval_loss = evaluate(args, model, eval_dataset)
                    print("Benchmark Eval:\n"
                          "Total: {}\n"
                          "Text: {}\n"
                          "Video: {}\n"
                          "Joint: {}\n".format(total_eval_loss, text_eval_loss, video_eval_loss, joint_eval_loss))
                    model.train()
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def main(colab_args=None):
    if colab_args:
        args = colab_args
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument(
            "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
        )
        parser.add_argument(
            "--train_data_path",
            default=None,
            type=str,
            help="The json file for training the model"
        )
        parser.add_argument(
            "--eval_data_path",
            default=None,
            type=str,
            help="The json file for evaluating the model"
        )
        parser.add_argument(
            "--config_name",
            default=None,
            type=str,
            help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
        )
        parser.add_argument(
            "--cache_dir",
            default=None,
            type=str,
            help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
        )
        parser.add_argument(
            "--block_size",
            default=-1,
            type=int,
            help="Optional input sequence length after tokenization."
                 "The training dataset will be truncated in block of this size for training."
                 "Default to the model max input length for single sentence inputs (take into account special tokens).",
        )
        parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
        )
        parser.add_argument(
            "--max_steps",
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        parser.add_argument("--log_dir", default=".", type=str, help="Directory to store the logs.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
        parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
        parser.add_argument(
            "--save_total_limit",
            type=int,
            default=None,
            help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
        )
        parser.add_argument(
            "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
        )
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
            and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0 if device == 'cpu' else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    # setup tokenizer and model
    if os.path.exists(os.path.join(args.output_dir, "tokenizer.pt")):
        new_tokenizer = False
        tokenizer = torch.load(os.path.join(args.output_dir, "tokenizer.pt"))
    else:
        new_tokenizer = True
        tokenizer = Field(tokenize=tokenize_en,
                          init_token='<sos>',
                          eos_token='<eos>',
                          lower=True,
                          batch_first=True)

    if args.model_name_or_path is None:
        # start from inital model
        print('### LOADING INITIAL MODEL ###')
        model = VideoTransformer(config=data_globals.config, args=args)
        model.apply(initialize_weights)
    else:
        # start from checkpoint
        print('### LOADING MODEL FROM CHECKPOINT:', args.model_name_or_path, '###')
        model = VideoTransformer.from_pretrained(config=data_globals.config, args=args)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = VideoBertDataset(tokenizer, build_tokenizer=new_tokenizer, data_path=args.train_data_path)
    eval_dataset = VideoBertDataset(train_dataset.tokenizer, build_tokenizer=False, data_path=args.eval_data_path)

    if new_tokenizer:
        torch.save(train_dataset.tokenizer, os.path.join(args.output_dir, "tokenizer.pt"))
        logger.info("Saving tokenizer to %s", args.output_dir)

    # Benchmark Evaluation
    total_avg_loss, text_avg_loss, video_avg_loss, joint_avg_loss = evaluate(args, model, eval_dataset)
    print("Benchmark Eval:\n"
          "Total: {}\n"
          "Text: {}\n"
          "Video: {}\n"
          "Joint: {}\n".format(total_avg_loss, text_avg_loss, video_avg_loss, joint_avg_loss))

    # Start Training
    model.train()
    global_step, tr_loss = train(args, model, train_dataset, eval_dataset)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
