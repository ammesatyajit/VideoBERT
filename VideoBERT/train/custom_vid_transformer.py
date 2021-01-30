import torch
from torch import nn

from VideoBERT.train.model_utils import contains_nan


class VideoTransformer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

        self.tok_embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_encoding = nn.Embedding(300, self.config.hidden_size)
        self.tok_type_embed = nn.Embedding(2, self.config.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.config.hidden_size])).to(self.args.device)
        assert len(self.scale.shape) == 1
        assert contains_nan(self.scale).item() is False

        self.fc_out = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        num_layers = self.config.num_hidden_layers//2
        self.transformer = nn.Transformer(d_model=self.config.hidden_size,
                                          nhead=self.config.num_attention_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          activation=self.config.hidden_act,
                                          dropout=0.1)

    def forward(
        self,
        text_input_ids=None,
        video_input_ids=None,
        joint_input_ids=None,

        text_token_type_ids=None,
        video_token_type_ids=None,
        joint_token_type_ids=None,

        text_attention_mask=None,
        video_attention_mask=None,
        joint_attention_mask=None,
    ):

        outputs = []
        text_loss = None
        video_loss = None
        joint_loss = None

        if text_input_ids is not None:
            text_mask = self._generate_square_subsequent_mask(text_input_ids.shape[1]-1).to(self.args.device)
            text_out = self.get_outputs(
                seq=text_input_ids[:, :-1].to(self.args.device),
                tok_type_ids=text_token_type_ids[:, :-1].to(self.args.device),
                attn_mask=text_mask,
                key_pad_mask=text_attention_mask[:, :-1].to(self.args.device),
            )

            outputs.append(text_out)

            loss_fct = torch.nn.CrossEntropyLoss()
            text_out = text_out.permute(2, 0, 1)
            text_out = text_out.view(self.config.vocab_size, -1).permute(1, 0)
            text_loss = loss_fct(text_out, text_input_ids[:, 1:].contiguous().view(-1).to(self.args.device))
            outputs.append(text_loss)

        if video_input_ids is not None:
            vid_mask = self._generate_square_subsequent_mask(video_input_ids.shape[1]-1).to(self.args.device)
            vid_out = self.get_outputs(
                seq=video_input_ids[:, :-1].to(self.args.device),
                tok_type_ids=video_token_type_ids[:, :-1].to(self.args.device),
                attn_mask=vid_mask,
                key_pad_mask=video_attention_mask[:, :-1].to(self.args.device),
            )

            outputs.append(vid_out)

            loss_fct = torch.nn.CrossEntropyLoss()
            vid_out = vid_out.permute(2, 0, 1)
            vid_out = vid_out.view(self.config.vocab_size, -1).permute(1, 0)
            video_loss = loss_fct(vid_out, video_input_ids[:, 1:].contiguous().view(-1).to(self.args.device))
            outputs.append(video_loss)

        if joint_input_ids is not None:
            joint_mask = self._generate_square_subsequent_mask(joint_input_ids.shape[1]-1).to(self.args.device)
            joint_out = self.get_outputs(
                seq=joint_input_ids[:, :-1].to(self.args.device),
                tok_type_ids=joint_token_type_ids[:, :-1].to(self.args.device),
                attn_mask=joint_mask,
                key_pad_mask=joint_attention_mask[:, :-1].to(self.args.device),
            )

            outputs.append(joint_out)

            loss_fct = torch.nn.CrossEntropyLoss()
            joint_out = joint_out.permute(2, 0, 1)
            joint_out = joint_out.view(self.config.vocab_size, -1).permute(1, 0)
            joint_loss = loss_fct(joint_out, joint_input_ids[:, 1:].contiguous().view(-1).to(self.args.device))
            outputs.append(joint_loss)

        if text_loss is not None and video_loss is not None and joint_loss is not None:
            total_loss = (text_loss + video_loss + joint_loss) / 3.0
            outputs = [total_loss, *outputs]

        return outputs

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_outputs(self, seq, tok_type_ids, attn_mask, key_pad_mask):
        pos = self.pos_encoding(torch.arange(0, seq.shape[1]).unsqueeze(0).repeat(seq.shape[0], 1).to(self.args.device))
        tok = self.tok_embed(seq) * self.scale
        tok_type = self.tok_type_embed(tok_type_ids)

        seq = self.dropout(tok + pos + tok_type)
        try:
            if contains_nan(pos):
                print(contains_nan(pos))
                raise RuntimeError("pos contains a nan")
            if contains_nan(tok):
                print(contains_nan(tok))
                raise RuntimeError("tok contains a nan")
            if contains_nan(tok_type):
                print(contains_nan(tok_type))
                raise RuntimeError("tok_type contains a nan")
        except:
            print(seq.shape[1])

        seq = seq.transpose(0, 1)

        out = self.transformer(seq,
                               seq,
                               src_mask=attn_mask,
                               tgt_mask=attn_mask,
                               memory_mask=attn_mask,
                               src_key_padding_mask=key_pad_mask,
                               tgt_key_padding_mask=key_pad_mask,
                               memory_key_padding_mask=key_pad_mask).transpose(0, 1)
        return self.fc_out(out)

    @classmethod
    def from_pretrained(cls, config, args):
        model = cls(config, args)
        model.load_state_dict(torch.load(args.model_name_or_path + '/pytorch_model.bin'))
        model.eval()
        return model

    def save_pretrained(self, output_dir):
        torch.save(self.state_dict(), output_dir + '/pytorch_model.bin')
