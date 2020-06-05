from transformers import BertConfig, BertTokenizer
import torch


bert_model = 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model)

bert_vocab_size = len(tokenizer)
visual_vocab_size = 20544
visual_linguistic_glue = 1

total_vocab_size = bert_vocab_size + \
                   visual_vocab_size + \
                   visual_linguistic_glue

vis_lin_glue_token_id = bert_vocab_size + visual_vocab_size

frozen_indices = torch.arange(bert_vocab_size, dtype=torch.int64)

data_path = 'data/newest-data-max-len-20.npy'
centers_file = 'data/centers.npy'

config = BertConfig(
    vocab_size=total_vocab_size,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
)
