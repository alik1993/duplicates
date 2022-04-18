import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class SentimentCNN(nn.Module):
    def __init__(self, embed_matrix, n_filters,
                 filter_sizes, output_dim, dropout, pad_idx, activation='sigmoid'):
        super().__init__()
#         Матрица эмбеддингов слов
        embed_matrix = torch.from_numpy(embed_matrix).float()
        self.embedding = self.embedding = nn.Embedding.from_pretrained(embed_matrix,
                                                                       freeze=True)
        
        embedding_dim = embed_matrix.shape[1]
#         Одномерные сверточные слои
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters,
                                             kernel_size=fs, padding=fs//2)
                                    for fs in filter_sizes
                                    ])
#         Линейный слой для понижения размерности
        output_dim, int_dim = 700, 300
        
        self.act = nn.ReLU() if activation != 'sigmoid' else nn.Sigmoid()
        self.fc = nn.Linear(len(filter_sizes) * n_filters * 2 + 13, output_dim)
        self.fc_act = nn.ReLU() if activation != 'sigmoid' else nn.Sigmoid()
#         Дропаут слой
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(len(filter_sizes) * n_filters * 2 + 13)
        self.batchnorm2 = nn.BatchNorm1d(output_dim)
        
        
#         self.int_fc = nn.Linear(output_dim, int_dim)
#         self.int_fc_act = nn.ReLU() if activation != 'sigmoid' else nn.Sigmoid()
#         self.int_batchnorm = nn.BatchNorm1d(int_dim)
#         self.int_dropout = nn.Dropout(dropout)
        
        self.fc_last = nn.Linear(output_dim, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward_(self, text):
#         Получаем эмбеддинги слов
        embedded = self.embedding(text)
#         Переставим размерности местами
        embedded = embedded.permute(0, 2, 1)
#         Применим 3 различных одномерных свертки с разными размерами ядер
        conved = [F.relu(conv(embedded)) for conv in self.convs]
#         Применим над каждым выходом сверточного слоя одномерный макс-пулинг
        pooled = [F.max_pool1d(conv, conv.size()[2]).squeeze(2) for conv in conved]
#         Сконкатенируем выходы с каждого пулинг слоя в один вектор и применим к нему дропаут
        cat = torch.cat(pooled, dim=1)
#         Пропустим итоговый вектор через полносвязный слой для понижения размерности
        return cat
    
    def forward(self, text1, text2, features):
#         Пропустим каждый текст через сверточную нейросеть для получения эмбеддингов этих текстов
#         в латентном пространстве для последующего сравнения по косинусному расстоянию
        x1 = self.forward_(text1)
        x2 = self.forward_(text2)
#         print(x1.size(), x2.size())
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
