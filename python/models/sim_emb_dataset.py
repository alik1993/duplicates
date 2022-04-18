import numpy as np

import zlib

from torch.utils.data import Dataset

import torch
import torch.nn.functional as F

import Levenshtein

import helpers
from models.tokenizer import tokenize


def create_features(sent1, sent2):
    wmd = helpers.ft_model.wv.wmdistance(sent1, sent2)
    wmd = wmd if wmd != np.inf else 10.0

    # почситаем сжатое представление текстов и их конкатенации
    zsent1 = zlib.compress(' '.join(sent1).encode('utf-8'))
    zsent2 = zlib.compress(' '.join(sent2).encode('utf-8'))
    zsents = zlib.compress(' '.join(sent1 + sent2).encode('utf-8'))

    # добавим различные числовые признаки текстов
    features = [len(sent1), # длина первого текста
                len(sent2), # длина второго текста
                len(sent1)/len(sent2), # частное длин текстов
                len(set(sent1).intersection(set(sent2))), # количество одинаковых слов в текстах
                len(set(sent1).union(set(sent2))), # количество всего разных слов в двух текстах
                # джакардова мера на множествах слов текстов
                len(set(sent1).intersection(set(sent2)))/len(set(sent1).union(set(sent2))),
                wmd, # word moving distance
                len(zsents), # длина сжатых строк
                len(zsent1), # длина сжатой первой строки
                len(zsent2), # длина сжатой второй строки
                # комбинации длин сжатых строк
                len(zsents) / max([len(zsent1), len(zsent2)]),
                len(zsents) / min([len(zsent1), len(zsent2)]),
                # расстояние Левенштейна между текстами
                Levenshtein.distance(' '.join(sent1), ' '.join(sent2))
                ]
    return features


# создадим класс для подачи данных нейросетям
class SimEmbDataset(Dataset):
    def __init__(self, df, TEXT):
        self.df = df
        self.TEXT = TEXT

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # отберем тексты из семпла по индексу i
        sent1 = tokenize(self.df.iloc[i]['content1'])[:200]
        sent2 = tokenize(self.df.iloc[i]['content2'])[:200]
        # переведем тексты в последовательность индексов из словаря
        itoks1 = [self.TEXT.vocab.stoi[x] for x in sent1]
        itoks2 = [self.TEXT.vocab.stoi[x] for x in sent2]
        # возьмем ответ на семпле по индексу i
        label = self.df.iloc[i]['answer']

        # определим размер паддинга
        n = 200
        pad1 = (0, n - len(itoks1))
        pad2 = (0, n - len(itoks2))

        # добавим различные числовые признаки текстов
        features = create_features(sent1, sent2)

        # возвращаем словарь с тензорами: тексты, длины текстов, числовые признаки, правильный ответ
        return {
            'text1': F.pad(torch.Tensor([itoks1]), pad1, 'constant', self.TEXT.vocab.stoi[self.TEXT.pad_token]).long().squeeze(0),
            'text2': F.pad(torch.Tensor([itoks2]), pad2, 'constant', self.TEXT.vocab.stoi[self.TEXT.pad_token]).long().squeeze(0),
            'length1': torch.Tensor([len(itoks1)]),
            'length2': torch.Tensor([len(itoks2)]),
            'features': torch.Tensor(features),
            'label': torch.Tensor([label]).squeeze(0)
        }
