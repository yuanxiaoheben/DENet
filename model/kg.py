import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F


class EvidenceEnsemble(nn.Module):
    def __init__(self, configs, word_embeddings):
        super(EvidenceEnsemble, self).__init__()
        emb_list = []
        for idx, window_size in enumerate([1, 2, 3, 5]):
            emb_list.append(ConvEmbedding(window_size, word_embeddings, configs.k_dim, \
            configs.dropout_prob, configs.per_channel))
        self.emb_list = nn.ModuleList(emb_list)
        self.dropout = nn.Dropout(p = configs.dropout_prob)
        # self.batch_size = batch_size 
        input_dim = sum([v.get_out_dim() for v in self.emb_list])
        self.lstm = torch.nn.LSTM(input_dim, configs.lstm_hid_dim, 2, batch_first=True, \
            bidirectional=True, dropout = configs.dropout_prob)
        self.lstm_hid_dim = configs.lstm_hid_dim
        self.mh_att = EvidenceAttentionEncoding(2 * configs.lstm_hid_dim, len(self.emb_list) * configs.evidence_dim, \
            configs.num_heads, configs.d_model)
        self.d_model = configs.d_model
        self.linear_final = torch.nn.Linear(configs.d_model, configs.num_labels)
    
    def forward(self, input_list):
        # Pass the input through your layers in order
        # attenton
        eb_s = []
        all_evidence = []
        for idx, data in enumerate(input_list):
            eb_s.append(self.emb_list[idx](data['w_ids'], data['k']))
            all_evidence.append(data['e'])
        outputs = torch.cat(eb_s, dim=2)
        evidence = torch.cat(all_evidence, dim=2)
        evidence = self.dropout(evidence)
        outputs, self.hidden_state = self.lstm(outputs) 
        att_out = self.mh_att(outputs, evidence)
        avg_sentence_embeddings = torch.sum(att_out, 1) / self.d_model
        output = self.linear_final(avg_sentence_embeddings)
        return output
    def _load_embeddings(self,embeddings):
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)
        return word_embeddings,emb_dim
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

class ConvEmbedding(nn.Module):
    def __init__(self, window_size, word_embeddings, knowlege_dim, drop_rate, per_channel):
        super(ConvEmbedding, self).__init__()
        self.word_emb, word_embedding_dim = self._load_embeddings(word_embeddings)
        kernels, channels = self.get_conv(window_size, per_channel)
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=word_embedding_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)
        self.knowlege_dim = knowlege_dim
        self.channels = channels

    def forward(self, sentence_input, knowlege):
        # sentence_input (batch_size, token_seq_len, window_seq_len, token_dim)
        # evidence (batch_size, token_seq_len, evidence_dim)
        token_emb = self.word_emb(sentence_input)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        token_emb = self.dropout(token_emb)
        knowlege = knowlege.permute(0, 2, 1) # (batch_size, k_dim, token_seq_len)
        token_emb = token_emb.permute(0, 3, 1, 2)  # (batch_size, token_dim, token_seq_len, window_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(token_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, token_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), token_seq_len)
        char_output = torch.cat((char_output, knowlege), dim=1) # (batch_size, sum(channels) + evidence_dim, token_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, token_seq_len, sum(channels) + evidence_dim)
    def _load_embeddings(self,embeddings):
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)
        return word_embeddings,emb_dim
    
    def get_conv(self, window_size, per_channel):
        kernels = [x for x in range(1, window_size + 1)]
        channels = [x * per_channel for x in kernels]
        return kernels, channels
    
    def get_out_dim(self):
        return sum(self.channels)  + self.knowlege_dim


class EvidenceAttentionEncoding(nn.Module):
    def __init__(self, text_input_dim, evidence_dim, h, d_model = 512):
        super(EvidenceAttentionEncoding, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.wq = nn.Linear(evidence_dim, d_model)
        self.wk = nn.Linear(text_input_dim, d_model)
        self.wv = nn.Linear(text_input_dim, d_model)
        self.linears = self._clones(nn.Linear(d_model, d_model), 4)

    
    def forward(self, seq_input, evidence):
        # Q, K ,V
        query = self.wq(evidence)
        key = self.wk(seq_input)
        value = self.wv(seq_input)
        nbatches = evidence.size(0)
        # Attention
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)	
				for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = F.softmax(scores, dim = -1)
        out = torch.matmul(p_attn, value)
        out = out.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # out_dim (batch, seq_len, d_model)
        return out
    
    def _load_embeddings(self,embeddings):
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)
        return word_embeddings,emb_dim
    def _clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])