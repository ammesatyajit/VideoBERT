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

    top_stats = {
        'verbs_top1': [],
        'verbs_top5': [],
        'nouns_top1': [],
        'nouns_top5': [],
    }

    npreds = 5

    predictmode = 'vid-prior'

    import json
    with open(data_globals.val_youcook, 'r') as fd:
        data = json.load(fd)

        for nr, (id, val) in enumerate(data.items(), start=1):
            print('nr:', nr)
            annots = val['annotations']

            for an in annots:
                template_sent = "now let me show you how to [MASK] the [MASK]."
                encoded = tokenizer.encode(template_sent, add_special_tokens=False)

                verbs_nouns_filt = an['verbs_nouns_filtered']
                verbs = verbs_nouns_filt['verbs']
                nouns = verbs_nouns_filt['nouns']
                vsent = an['video_ids']

                vid_template = tokenizer.encode("[MASK] [MASK] [MASK] [MASK]", add_special_tokens=False)
                print("vsent:", np.array(vsent)+30522)

                if len(vsent) > 0 and (len(verbs) > 0 or len(nouns) > 0):
                    if predictmode == 'lang-prior':
                        input_ids = torch.tensor(np.hstack([
                            np.array([101]),
                            np.array(encoded),
                            np.array([102])
                        ]), dtype=torch.int64).unsqueeze(0)
                    elif predictmode == 'vid-prior':
                        input_ids = torch.tensor(np.hstack([
                            np.array([101]),
                            np.array(vsent)[0] + 30522,
                            np.array(vid_template),
                            np.array([102])
                        ]), dtype=torch.int64).unsqueeze(0)
                    else:
                        input_ids = torch.tensor(np.hstack([
                            np.array([101]),
                            np.array(encoded),
                            np.array([30522 + 20544]),
                            np.array(vsent) + 30522,
                            np.array([102])
                        ]), dtype=torch.int64).unsqueeze(0)

                    print(input_ids)
                    quit()

                    input_ids = input_ids.to(device)

                    token_type_ids = torch.tensor(np.hstack([
                        np.zeros(len(encoded) + 2),
                        np.ones(len(vsent) + 1)
                    ]), dtype=torch.int64)

                    token_type_ids = token_type_ids.to(device)

                    if predictmode == 'lang-prior':
                        outputs = model(
                            text_input_ids=input_ids,
                        )
                    elif predictmode == 'vid-prior':
                        outputs = model(
                            video_input_ids=input_ids
                        )
                    else:
                        outputs = model(
                            joint_input_ids=input_ids,
                            joint_token_type_ids=token_type_ids,
                        )

                    input_ids = input_ids.to(torch.device('cpu'))

                    input_ids = input_ids[0]

                    prediction_scores = outputs[0]

                    mask_indices = (input_ids == tokenizer.mask_token_id).nonzero().squeeze()

                    if mask_indices.dim() == 0:
                        mask_indices = torch.tensor([mask_indices])

                    if type(mask_indices) == int:
                        mask_indices = torch.tensor([mask_indices])

                    results = [[], []]

                    for index, masked_index in enumerate(mask_indices):
                        # print('mask index:', masked_index)
                        logits = prediction_scores[0, masked_index, :]
                        probs = logits.softmax(dim=0)
                        values, predictions = probs.topk(npreds)

                        for i, (v, p) in enumerate(zip(values.tolist(), predictions.tolist())):
                            results[index].append(tokenizer.decode([p]))

                    predicted_verbs = results[0][:5]
                    if predictmode == 'lang-prior':
                        predicted_nouns = results[1][:5]
                    else:
                        predicted_nouns = results[1][1:]

                    print('verbs:', verbs)
                    print('predicted verbs:', predicted_verbs)

                    print('nouns:', nouns)
                    print('predicted nouns:', predicted_nouns)
                    print('')

                    if len(verbs) > 0:
                        top_stats['verbs_top1'].append(predicted_verbs[0] in verbs)

                    if len(nouns) > 0:
                        top_stats['nouns_top1'].append(predicted_nouns[0] in nouns)

                    if len(verbs) > 0:
                        verbs_top5 = False

                        for v in predicted_verbs:
                            if v in verbs:
                                verbs_top5 = True
                                break

                        top_stats['verbs_top5'].append(verbs_top5)

                    if len(nouns) > 0:
                        nouns_top5 = False

                        for n in predicted_nouns:
                            if n in nouns:
                                nouns_top5 = True
                                break

                        top_stats['nouns_top5'].append(nouns_top5)

    verbs_top1 = np.array(top_stats['verbs_top1'])
    verbs_top5 = np.array(top_stats['verbs_top5'])

    nouns_top1 = np.array(top_stats['nouns_top1'])
    nouns_top5 = np.array(top_stats['nouns_top5'])

    verbs_top1_acc = len(verbs_top1.nonzero()[0]) / len(verbs_top1)
    verbs_top5_acc = len(verbs_top5.nonzero()[0]) / len(verbs_top5)

    nouns_top1_acc = len(nouns_top1.nonzero()[0]) / len(nouns_top1)
    nouns_top5_acc = len(nouns_top5.nonzero()[0]) / len(nouns_top5)

    print('verbs top 1:', verbs_top1_acc)
    print('verbs top 5:', verbs_top5_acc)

    print('nouns top 1:', nouns_top1_acc)
    print('nouns top 5:', nouns_top5_acc)


if __name__ == "__main__":
    main()

