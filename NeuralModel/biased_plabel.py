"""
Definition of the ESIM model.
"""

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, BiasedAttention
from .utils import get_mask, replace_masked


class ESIM(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention2 = SoftmaxAttention()
        
        self._projection = nn.Sequential(nn.Linear(5*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*6*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))
        self.pls = nn.Linear(2*self.hidden_size, 2)
        self.apply(_init_esim_weights)
        
        self._attention = BiasedAttention()
    def forward(self,
                premises,
                premises_lengths,
                premises2,
                premises_lengths2,
                hypotheses,
                hypotheses_lengths,
                assert_test2_matrix,
                test1_test2_matrix
            ):
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        premises_mask2 = get_mask(premises2, premises_lengths2).to(self.device)
        
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_premises2 = self._word_embedding(premises2)
        
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_premises2 = self._rnn_dropout(embedded_premises2)
            
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_premises2 = self._encoding(embedded_premises2,
                                          premises_lengths2)
        
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)
        
        attended_hypotheses, attended_premises, plogits=\
            self._attention(encoded_hypotheses, hypotheses_mask, 
                            encoded_premises, premises_mask,assert_test2_matrix, True)
        
        attended_premises2, attended_hypotheses2 =\
            self._attention2(encoded_premises2, premises_mask2,
                            encoded_hypotheses, hypotheses_mask)
        attended_premises2_x, attended_premises_x, pplogits=\
            self._attention(encoded_premises2, premises_mask2,
                            encoded_premises, premises_mask, test1_test2_matrix, True)
        

        enhanced_premises = torch.cat([encoded_premises,
                                            attended_premises,
                                            attended_premises_x,
                                            encoded_premises - attended_premises,
                                            encoded_premises * attended_premises],
                                            dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         attended_hypotheses2,
                                         attended_hypotheses -
                                         attended_hypotheses2,
                                         attended_hypotheses *
                                         attended_hypotheses2],
                                        dim=-1)
        
        enhanced_premises2 = torch.cat([encoded_premises2,
                                       attended_premises2,
                                       attended_premises2_x,
                                       encoded_premises2 - attended_premises2,
                                       encoded_premises2 * attended_premises2],
                                      dim=-1)
        projected_premises = self._projection(enhanced_premises)
        projected_premises2 = self._projection(enhanced_premises2)
        projected_hypotheses = self._projection(enhanced_hypotheses)
        
        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_premises2 = self._rnn_dropout(projected_premises2)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_ai2 = self._composition(projected_premises2, premises_lengths2)

        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        #plogits = self.pls(v_bj)
        
        #return plogits
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_a_avg2 = torch.sum(v_ai2 * premises_mask2.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask2, dim=1, keepdim=True)
        
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_a_max2, _ = replace_masked(v_ai2, premises_mask2, -1e7).max(dim=1)
        
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_avg2, v_a_max, v_a_max2, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return plogits, pplogits, logits, probabilities


def _init_esim_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
