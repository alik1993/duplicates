import torch as tt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from operator import itemgetter

class Attention(nn.Module):
    def __init__(self, feature_size, attn_size):
        super().__init__()
        self.feature_size = feature_size
        self.W = nn.Linear( self.feature_size, attn_size)
        self.V = nn.Linear(attn_size, 1, bias=False)
        
    def forward(self, x, mask=None):        
        batch_size, seq_length = x.size(0), x.size(1)
        e_ij  = self.W(x.contiguous().view(-1, self.feature_size))
        e_ij = tt.tanh(e_ij)
        e_ij = self.V(e_ij)
        e_ij = e_ij.contiguous().view(batch_size, seq_length)
        
        attn = F.softmax(e_ij, dim=1)        
        
        weighted_x = x * attn.unsqueeze(2)
        
        return weighted_x.sum(dim=1)
    
class SentimentModel(nn.Module):
    
    def __init__(self, embed_matrix, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout):
        
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, 
                                                      freeze=True)
        self.rnn1 = nn.LSTM(input_size=embed_matrix.size(1), 
                           hidden_size=hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True,
                           dropout=dropout)
        
        self.attn1 = Attention(hidden_dim * (2 if bidirectional else 1), hidden_dim // 4)
        
#         self.rnn2 = nn.LSTM(input_size=embed_matrix.size(1), 
#                            hidden_size=hidden_dim, 
#                            num_layers=n_layers, 
#                            bidirectional=bidirectional, 
#                            batch_first=True,
#                            dropout=dropout)
        
#         self.attn2 = Attention(hidden_dim * (2 if bidirectional else 1), hidden_dim // 4)
        
        fc_size = hidden_dim * (2 if bidirectional else 1) * (1 + 2 * n_layers)
        
        self.fc1 = nn.Linear(fc_size * 2 + 13, 1)
        self.act = nn.Sigmoid()
#         self.fc2 = nn.Linear(fc_size, output_dim)
        
        self.dropout1_1 = nn.Dropout(dropout / 2)
        self.dropout1_2 = nn.Dropout(dropout / 2)
        self.dropout1_3 = nn.Dropout(dropout / 2)
        
        self.dropout2_1 = nn.Dropout(dropout / 2)
        self.dropout2_2 = nn.Dropout(dropout / 2)
        self.dropout2_3 = nn.Dropout(dropout / 2)
        
        output_dim == 300
        
        self.act = nn.ReLU()
        self.fc = nn.Linear(fc_size * 2 + 13, output_dim)
        self.fc_act = nn.ReLU()
#         Дропаут слой
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(fc_size * 2 + 13)
        self.batchnorm2 = nn.BatchNorm1d(output_dim)
        
        self.fc_last = nn.Linear(output_dim, 1)
        
        self.activation = nn.Sigmoid()
        
    def init_weights(self):
        
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
        
        
    def forward_(self, x1, x1_lengths):
#         Получаем эмбеддинги слов
        x1 = self.embedding(x1)
#         print(x1.size())
        x1 = self.dropout1_1(x1)
#         print(x1.size())
        total_length1 = x1.size(1)
#         print(x1_lengths.size())

        x1_lengths = x1_lengths.view(-1).tolist()
        ss = sorted(zip(x1_lengths, range(len(x1_lengths)), x1), key=itemgetter(0), reverse=True)
        orig_order = [x[1] for x in ss]
        x1 = tt.cat([x[2].unsqueeze(0) for x in ss], dim=0)
        x1 = pack_padded_sequence(x1, sorted(x1_lengths, reverse=True), batch_first=True)
        
        self.rnn1.flatten_parameters()
        packed_output1, (hidden1, cell1) = self.rnn1(x1)
        output1, _ = nn.utils.rnn.pad_packed_sequence(packed_output1, total_length=total_length1, batch_first=True)
        output1 = self.dropout1_2(output1)
        x1 = self.attn1(output1)
#         print(x1.size())
        
        hidden1 = hidden1.transpose(0, 1)
        hidden1 = hidden1.contiguous().view(hidden1.size(0), -1)
        cell1 = cell1.transpose(0, 1)
        cell1 = cell1.contiguous().view(cell1.size(0), -1)

        x1 = tt.cat([x1, hidden1, cell1], dim=1)
#         x1 = self.dropout1_3(x1)
#         x1 = self.fc1(x1).squeeze(1)

        sr = sorted(zip(orig_order, x1), key=itemgetter(0))
        x1 = tt.cat([x[1].unsqueeze(0) for x in sr], dim=0)
#         Пропустим итоговый вектор через полносвязный слой для понижения размерности
        return x1
        
    
    def forward(self, x1, x2, features, x1_lengths, x2_lengths):
#         print(x1.size(), x2.size())

        
        x1 = self.forward_(x1, x1_lengths)
        x2 = self.forward_(x2, x2_lengths)
#         x1 = self.embedding(x1)
# #         print(x1.size())
#         x1 = self.dropout1_1(x1)
# #         print(x1.size())
#         total_length1 = x1.size(1)
# #         print(x1_lengths.size())
#         if x1_lengths is not None:
#             x1_lengths = x1_lengths.view(-1).tolist()
#             ss = sorted(zip(x1_lengths, range(len(x1_lengths)), x1), key=itemgetter(0), reverse=True)
#             orig_order = [x[1] for x in ss]
#             x1 = tt.cat([x[2].unsqueeze(0) for x in ss], dim=0)
#             x1 = nn.utils.rnn.pack_padded_sequence(x1, sorted(x1_lengths, reverse=True), batch_first=True)
        
#         self.rnn1.flatten_parameters()
#         packed_output1, (hidden1, cell1) = self.rnn1(x1)
#         output1, _ = nn.utils.rnn.pad_packed_sequence(packed_output1, total_length=total_length1, batch_first=True)
#         output1 = self.dropout1_2(output1)
#         x1 = self.attn1(output1)
# #         print(x1.size())
        
#         hidden1 = hidden1.transpose(0, 1)
#         hidden1 = hidden1.contiguous().view(hidden1.size(0), -1)
#         cell1 = cell1.transpose(0,1)
#         cell1 = cell1.contiguous().view(cell1.size(0), -1)

#         x1 = tt.cat([x1, hidden1, cell1], dim=1)
        

    
        ## x2
#         x2 = self.embedding(x2)
#         x2 = self.dropout1_1(x2)

#         total_length2 = x2.size(1)
        
#         if x2_lengths is not None:
#             x2_lengths = x2_lengths.view(-1).tolist()
#             ss = sorted(zip(x2_lengths, range(len(x2_lengths)), x2), key=itemgetter(0), reverse=True)
#             orig_order = [x[1] for x in ss]
#             x2 = tt.cat([x[2].unsqueeze(0) for x in ss], dim=0)
#             x2 = nn.utils.rnn.pack_padded_sequence(x2, sorted(x2_lengths, reverse=True), batch_first=True)
        
#         self.rnn1.flatten_parameters()
#         packed_output2, (hidden2, cell2) = self.rnn1(x2)
#         output2, _ = nn.utils.rnn.pad_packed_sequence(packed_output2, total_length=total_length2, batch_first=True)
#         output2 = self.dropout1_2(output2)
#         x2 = self.attn1(output2)
    
#         hidden2 = hidden2.transpose(0, 1)
#         hidden2 = hidden2.contiguous().view(hidden2.size(0), -1)
#         cell2 = cell2.transpose(0,1)
#         cell2 = cell2.contiguous().view(cell2.size(0), -1)

#         x2 = tt.cat([x2, hidden2, cell2], dim=1)
        x = torch.cat([x1, x2, features], dim=1)
        
        x = self.batchnorm1(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        x = self.fc(x)
        x = self.batchnorm2(x)
        x = self.fc_act(x)
        x = self.dropout2(x)
        
#         x = self.int_fc(x)
#         x = self.int_batchnorm(x)
#         x = self.int_fc_act(x)
#         x = self.int_dropout(x)
        
        
#         print(x)
        return self.activation(self.fc_last(x).squeeze(1))