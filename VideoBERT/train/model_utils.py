import torch
from transformers import BertForPreTraining
import VideoBERT.data.globals as data_globals


def get_bert_save_dict():
    import os

    state_path = data_globals.root_path + '/bert-large.pt'

    if os.path.exists(state_path):
        state = torch.load(state_path)
    else:
        model = BertForPreTraining.from_pretrained(data_globals.bert_model)
        state = model.state_dict()
        # cache state
        torch.save(state, state_path)
    return state


def create_video_bert_save_dict_from_bert(config, centers_file=data_globals.centers_file):
    import numpy as np

    centroids = torch.tensor(np.load(centers_file), dtype=torch.float32)
    n_extra_embeddings = centroids.shape[0] + 1

    bert_state = get_bert_save_dict()

    glue_embedding = torch.empty(1, config.hidden_size, dtype=torch.float32)
    torch.nn.init.normal_(glue_embedding, mean=0.0, std=config.initializer_range)

    word_embeddings = bert_state['bert.embeddings.word_embeddings.weight']
    word_embeddings = torch.cat([word_embeddings, centroids, glue_embedding], dim=0)

    cls_decoder_weight = bert_state['cls.predictions.decoder.weight']
    cls_decoder_weight = torch.cat([cls_decoder_weight, centroids, glue_embedding], dim=0)

    cls_decoder_bias = bert_state['cls.predictions.decoder.bias']
    extra_cls_decoder_bias = torch.empty(n_extra_embeddings, dtype=torch.float32)
    torch.nn.init.normal_(extra_cls_decoder_bias, mean=0.0, std=config.initializer_range)
    cls_decoder_bias = torch.cat([cls_decoder_bias, extra_cls_decoder_bias], dim=0)

    cls_predictions_bias = bert_state['cls.predictions.bias']
    extra_cls_predictions_bias = torch.empty(n_extra_embeddings, dtype=torch.float32)
    torch.nn.init.normal_(extra_cls_predictions_bias, mean=0.0,  std=config.initializer_range)
    cls_predictions_bias = torch.cat([cls_predictions_bias, extra_cls_predictions_bias], dim=0)

    vis_lin_align_weight = torch.empty(2, config.hidden_size, dtype=torch.float32)
    torch.nn.init.normal_(vis_lin_align_weight, mean=0.0, std=config.initializer_range)

    cls_vis_lin_align_bias = torch.empty(2, dtype=torch.float32)
    torch.nn.init.normal_(cls_vis_lin_align_bias, mean=0.0, std=config.initializer_range)

    video_bert_state = bert_state
    video_bert_state['bert.embeddings.word_embeddings.weight'] = word_embeddings
    video_bert_state['cls.predictions.decoder.weight'] = cls_decoder_weight
    video_bert_state['cls.predictions.decoder.bias'] = cls_decoder_bias
    video_bert_state['cls.predictions.bias'] = cls_predictions_bias
    video_bert_state['cls.vis_lin_align.weight'] = vis_lin_align_weight
    video_bert_state['cls.vis_lin_align.bias'] = cls_vis_lin_align_bias

    return video_bert_state
