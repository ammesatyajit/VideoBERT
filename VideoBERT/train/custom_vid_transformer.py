from transformers import BertForPreTraining, BertModel, BertForMaskedLM
from transformers.modeling_bert import BertPreTrainingHeads, BertOnlyMLMHead
import torch
from torch import nn


class VideoBertOnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.vis_lin_align = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, joint=False):
        prediction_scores = self.predictions(sequence_output)
        if joint:
            first_token_tensor = sequence_output[:, 0]  # take hidden state of [CLS] token
            vis_lin_align_score = self.vis_lin_align(first_token_tensor)
            return prediction_scores, vis_lin_align_score
        else:

            return prediction_scores


class VideoBertPreTrainingHeads(BertPreTrainingHeads):
    def __init__(self, config):
        super().__init__(config)
        self.vis_lin_align = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, joint=False):
        prediction_scores = self.predictions(sequence_output)

        if joint:
            first_token_tensor = sequence_output[:, 0]  # take hidden state of [CLS] token
            vis_lin_align_score = self.vis_lin_align(first_token_tensor)
            return prediction_scores, vis_lin_align_score
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
            return prediction_scores, seq_relationship_score


class VideoBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.cls = VideoBertPreTrainingHeads(config)

    def get_bert_outputs(self, input_ids, attention_mask, token_type_ids, joint=False):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output, pooled_output = outputs[:2]
        return self.cls(sequence_output, pooled_output, joint)

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

        text_masked_lm_labels=None,
        video_masked_lm_labels=None,
        joint_masked_lm_labels=None,

        text_next_sentence_label=None,
        video_next_sentence_label=None,
        joint_vis_lin_label=None,
    ):
        outputs = ()
        text_loss = None
        video_loss = None
        joint_loss = None

        if text_input_ids is not None:
            text_prediction_scores, text_seq_relationship_score = self.get_bert_outputs(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
            )

            outputs += (text_prediction_scores, text_seq_relationship_score)

            if text_masked_lm_labels is not None and text_next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(text_prediction_scores.view(-1, self.config.vocab_size), text_masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(text_seq_relationship_score.view(-1, 2), text_next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                text_loss = total_loss

        if video_input_ids is not None:
            video_prediction_scores, video_seq_relationship_score = self.get_bert_outputs(
                input_ids=video_input_ids,
                attention_mask=video_attention_mask,
                token_type_ids=video_token_type_ids,
            )

            outputs += (video_prediction_scores, video_seq_relationship_score)

            if video_masked_lm_labels is not None and video_next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(video_prediction_scores.view(-1, self.config.vocab_size), video_masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(video_seq_relationship_score.view(-1, 2), video_next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                video_loss = total_loss

        if joint_input_ids is not None:
            joint_prediction_scores, joint_vis_lin_score = self.get_bert_outputs(
                input_ids=joint_input_ids,
                attention_mask=joint_attention_mask,
                token_type_ids=joint_token_type_ids,
                joint=True  # joint mode
            )

            outputs += (joint_prediction_scores, joint_vis_lin_score)

            if joint_masked_lm_labels is not None and joint_vis_lin_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(joint_prediction_scores.view(-1, self.config.vocab_size), joint_masked_lm_labels.view(-1))
                vis_lin_align_loss = loss_fct(joint_vis_lin_score.view(-1, 2), joint_vis_lin_label.view(-1))
                total_loss = masked_lm_loss + vis_lin_align_loss
                joint_loss = total_loss

        if text_loss is not None and video_loss is not None and joint_loss is not None:
            total_loss = (text_loss + video_loss + joint_loss) / 3.0
            outputs = (total_loss, text_loss, video_loss, joint_loss,) + outputs

        return outputs


class VideoTransformer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

        self.tok_embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.pos_encoding = nn.Embedding(100, self.config.hidden_size)
        self.tok_type_embed = nn.Embedding(2, self.config.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor(self.config.hidden_size)).to(self.args.device)

        self.fc_out = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        self.transformer = nn.Transformer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads, activation=self.config.hidden_act)

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

        outputs = ()
        text_loss = None
        video_loss = None
        joint_loss = None

        if text_input_ids is not None:
            text_mask = self._generate_square_subsequent_mask(text_input_ids.shape[1]-1).to(self.args.device)
            text_out = self.get_outputs(
                seq=text_input_ids[:, :-1],
                tok_type_ids=text_token_type_ids[:, :-1],
                attn_mask=text_mask,
                key_pad_mask=text_attention_mask[:, :-1],
            )

            outputs += text_out

            if self.args.do_train:
                loss_fct = torch.nn.CrossEntropyLoss()
                text_loss = loss_fct(text_out.view(-1, self.config.vocab_size), text_input_ids[:, 1:].view(-1))

        if video_input_ids is not None:
            vid_mask = self._generate_square_subsequent_mask(video_input_ids.shape[1]-1).to(self.args.device)
            vid_out = self.get_outputs(
                seq=video_input_ids[:, :-1],
                tok_type_ids=video_token_type_ids[:, :-1],
                attn_mask=vid_mask,
                key_pad_mask=video_attention_mask[:, :-1],
            )

            outputs += vid_out

            if self.args.do_train:
                loss_fct = torch.nn.CrossEntropyLoss()
                text_loss = loss_fct(vid_out.view(-1, self.config.vocab_size), video_input_ids[:, 1:].view(-1))

        if joint_input_ids is not None:
            joint_mask = self._generate_square_subsequent_mask(joint_input_ids.shape[1]-1).to(self.args.device)
            joint_out = self.get_outputs(
                seq=joint_input_ids[:, :-1],
                tok_type_ids=joint_token_type_ids[:, :-1],
                attn_mask=joint_mask,
                key_pad_mask=joint_attention_mask[:, :-1],
            )

            outputs += joint_out

            if self.args.do_train:
                loss_fct = torch.nn.CrossEntropyLoss()
                text_loss = loss_fct(joint_out.view(-1, self.config.vocab_size), joint_input_ids[:, 1:].view(-1))

        if text_loss is not None and video_loss is not None and joint_loss is not None:
            total_loss = (text_loss + video_loss + joint_loss) / 3.0
            outputs = (total_loss, text_loss, video_loss, joint_loss,) + outputs

        return outputs

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_outputs(self, seq, tok_type_ids, attn_mask, key_pad_mask):
        pos = torch.arange(0, seq.shape[1]).unsqueeze(0).repeat(seq.shape[0], 1).to(self.args.device)
        seq = (self.tok_embed(seq) * self.scale) + self.pos_encoding(pos) + self.tok_type_embed(tok_type_ids)
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

    def from_pretrained(self, config, args):
        model = VideoTransformer(config, args)
        model.load_state_dict(torch.load(args.model_name_or_path + '/pytorch_model.bin'))
        return model
