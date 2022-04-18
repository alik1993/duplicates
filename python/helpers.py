import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from gensim.models.fasttext import FastText

import app_env

ft_model = FastText.load(app_env.data_model_builtin_path('bank_fasttextskipgram_300'))

def expand(dups):
    # копируем оригинальный датасет
    dups_expanded1 = dups.copy()

    dups_expanded1['content1'] = dups['content2']
    dups_expanded1['content2'] = dups['content1']

    if 'id1' in dups.columns and 'id2' in dups.columns:
        dups_expanded1['id1'] = dups['id2']
        dups_expanded1['id2'] = dups['id1']

    return pd.concat([dups, dups_expanded1], ignore_index=True)


# зададим функцию, которая считает accuracy для тензоров pytorch
def acc_binary(y_pred, y_true):
    y_pred = y_pred.numpy().ravel()
    y_true = y_true.numpy().ravel()
    return accuracy_score(y_true, (np.abs(y_pred) > 0.5).astype(int))


# зададим функцию, которая считает порог, который дает максимальное значение метрики accuracy на ответах
def max_th(y_true, y_pred):
    accs = dict()
    for th in np.linspace(0, 1, 50):
        accs[th] = accuracy_score(y_true, (y_pred>th).astype(int))

    return max(accs, key=lambda x: accs[x])
