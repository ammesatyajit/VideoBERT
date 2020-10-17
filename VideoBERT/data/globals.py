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

root_path = '/content/drive/My Drive/VideoBERT'
data_path = '/content/drive/My Drive/VideoBERT/training_data.csv'
centers_file = '/content/drive/My Drive/VideoBERT/centers.npy'
val_youcook = '/content/drive/My Drive/VideoBERT/val_data.csv'

config = BertConfig(
    vocab_size=total_vocab_size,
    hidden_size=1024,
    num_hidden_layers=16,
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
