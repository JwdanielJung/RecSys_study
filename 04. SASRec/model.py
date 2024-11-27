import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()

        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.linear2(self.relu(self.dropout1(self.linear1(inputs))))
        )
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()  # torch.nn.Module 구현위함

        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )  # padding 위해 item 갯수 + 1

        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )  # learnable positional embedding 위해 maxlen+1만큼 만들기

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # torch.nn.ModuleList() 통해 여러 layer 관리 가능 (nn.Linear 같은거 Module List에 append 하면 됨)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # multi-head attention을 num_block 만큼 쌓기
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feat(self, log_seqs):
        seqs = self.item_emb(
            torch.LongTensor(log_seqs).to(self.device)
        )  # [batch_size, sequence_length, hidden_units]
        seqs *= (
            self.item_emb.embedding_dim**0.5
        )  # 무슨의미? -> attention is all you need: 학습 안정성 위해 embedding layer에 d_model의 root 곱해줌 (output 차원의 제곱근 곱해줌)

        poss = np.tile(
            np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1]
        )  # log_seqs: [batch_size, sequence_length], sequence position 담기 위해 초기화

        poss *= (
            log_seqs != 0
        )  # 무슨의미? -> 0이면 패딩이므로, 패딩위치에 대해서 masking
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.device))
        seqs = self.emb_dropout(seqs)  # 논문에 embedding도 적용한다 써있음

        attention_mask = ~torch.tril(
            torch.ones(
                (seqs.shape[1], seqs.shape[1]), dtype=torch.bool, device=self.device
            )
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(
                seqs, 0, 1
            )  # sequence_length, batch_size, hidden_units
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )  # multiheadattention input으로 Query, key, value & attention_mask)
            seqs = Q + mha_outputs  # skip connection
            seqs = torch.transpose(
                seqs, 0, 1
            )  # 원상복귀 batch_size, sequence_length, hidden_units
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)  # Fc-layer 통과

        log_feats = self.last_layernorm(
            seqs
        )  # 기존 seqs tensor와 차원 변화x [batch_size, sequence_length, hidden_units]

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feat(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # elementwise 곱 이후 가장 마지막 차원 기준 summation -> pos_item의 점수
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # neg item의 점수

        return pos_logits, neg_logits


    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feat(log_seqs)

        final_feat = log_feats[:,-1,:] # 가장 마지막 => [batch_size,hidden_dims]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device)) # [batch_size, sequence_length, hidden_dims]

        # unsqueeze 통해 [batch_size,hidden_dims,1] 만듦 
        # => matmul은 마지막 두차원끼리 곱해짐 [sequence_length, hidden_dims] * [hidden_dims,1] => [batch_size, sequence_length,1]
        # squueze로 마지막 1차원 날림 => [batch_size, sequence_length]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) 

        return logits