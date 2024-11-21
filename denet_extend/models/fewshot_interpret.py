import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, out_size, dropout_prob, word_embeddings= None, evi_dim = 1232):
        super(BiLSTM, self).__init__()
        if not word_embeddings is None:
            self.word_embeds,self.word_embedding_dim = self._load_embeddings(word_embeddings)
        else:
            self.word_embedding_dim = embedding_dim
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim + evi_dim, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, evi):
        emb = self.word_embeds(sents_tensor)  # [B, L, emb_size]
        emb = torch.cat((emb, evi), dim=2)
        emb = self.dropout(emb)
        rnn_out, _ = self.bilstm(emb)
        # rnn_out:[B, L, hidden_size*2]

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, evi, lengths):
        logits = self.forward(sents_tensor, evi)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)
        out_data = []
        for seq_tags, seq_len in zip(batch_tagids, lengths):
            seq_tags = seq_tags.cpu().tolist()
            out_data.append(seq_tags[:int(seq_len)])
        return out_data
    
    def _load_embeddings(self,embeddings):
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)
        return word_embeddings,emb_dim