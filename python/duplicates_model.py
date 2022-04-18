import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import torchtext as tt

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

from models.sim_emb_dataset import SimEmbDataset

# загрузим нейросетевые модели
from nns.lstm import SentimentModel
from nns.gru import SentimentGRU
from nns.cnn import SentimentCNN
# загрузим класс для обучения нейросетей
from nns.trainer import TorchTrain

import app_env
import helpers
from models.tokenizer import tokenize

import os
import pathlib

SEED = 42


class DuplicatesModels:

    def __init__(self):
        # создадим объект для оборачивания текста
        self.TEXT = tt.data.Field(
            sequential=True,
            use_vocab=True,
            include_lengths=True,
            tokenize=tokenize,
            preprocessing = lambda x: x[: min(200, len(x))],
        )
        # создание объекта для работы с правильными ответами
        self.LABEL = tt.data.LabelField(
            use_vocab=False,
            dtype=torch.float32,
        )
        self.n_models = 3# число моделей: 3 нейросети
        self.n_folds = 5
        # девайс, на котором будут обучаться модели (GPU)
        if app_env.ml_params_use_gpu() is False:
            self.device = 'cpu'
        else:
            self.device = app_env.ml_params_cuda_devise()
        self.models = {}

    # метод для обучения всех моделей на обучающей выборке
    def train(self, df):
        # расширим наш датасет перестановкой текстов
        df = df.copy()
        print('Train size: ', df.shape)
        # зададим объекты для создания словаря токенов

        examples = [
            tt.data.Example.fromlist(
                x,
                [
                    ('label', self.LABEL),
                    ('content1', self.TEXT),
                    ('content2', self.TEXT),
                ]
            ) for _, x in df[['answer', 'content1', 'content2']].iterrows()
        ]

        text_data = tt.data.Dataset(
            examples,
            fields=[
                ('label', self.LABEL),
                ('content1', self.TEXT),
                ('content2', self.TEXT)
            ]
        )
        # создадим словарь
        self.TEXT.build_vocab(text_data, min_freq=50)
        # создадим матрицу эмбеддингов для слов из обучающей выборки
        self.embed_matrix = np.vstack(
            [
                np.random.randn(2, 300) * 0.25,
                np.vstack(
                    [
                        helpers.ft_model.wv[x] \
                            if (x in helpers.ft_model.wv) else np.random.randn(1, 300) * 0.25 \
                        for x in self.TEXT.vocab.itos[2:]
                    ]
                )
            ]
        )
        # зададим параметры для процесса обучения нейросетей:
        BATCH_SIZE = 512 # количество примеров, передающихся нейросети за один раз (батч)
        N_EPOCHS = 50 # количество обучений на одних данных
        WD = .1 # параметр регуляризации
        ES = 5 # количество эпох, после которых происходит остановка обучения, если качество не улучшается
        device = self.device
        # зададим матрицу для предсказаний out-of-fold
        oof_train = np.zeros((df.shape[0], self.n_models))
        # модели будем обучать на 5 фолдах

        # создадим структуру для сохранения обученных моделей
        self.models = {
            'lstm': [],
            'gru': [],
            'cnn': [],
        }
        # обучим в цикле по фолдам со стратифицированным разбиением данных
        stratify_cv = StratifiedKFold(self.n_folds, shuffle=True, random_state=SEED)
        for fold, (idx_train, idx_test) in enumerate(stratify_cv.split(df, df['answer'])):
            print('FOLD - {fold}'.format(fold=fold))

            # создадим объекты для передачи данных нейросети
            train = SimEmbDataset(helpers.expand(df.iloc[idx_train]), self.TEXT)
            test = SimEmbDataset(df.iloc[idx_test], self.TEXT)

            # создадим объекты для создания батчей
            train_iterator = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                                                         num_workers=app_env.ml_params_data_loader_num_workers())
            test_iterator = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False,
                                                        num_workers=app_env.ml_params_data_loader_num_workers())

            print('Start cnn')
            # создадим модель сверточной нейросети
            cnn = SentimentCNN(
                self.embed_matrix,
                n_filters=200,
                filter_sizes=[3,5,7],
                output_dim=150,
                dropout=0.5,
                pad_idx=self.TEXT.vocab.stoi[self.TEXT.pad_token],
            )
            # переместим ее на заданный девайс
            cnn.to(device)

            # зададим алгоритм обучения сверточной нейросети
            # зададим планировщик изменения параметра скорости обучения нейросети по экспоненциальному закону
            # выберем функцию ошибок - бинарную кросс-энтропию
            # создадим объект для обучения сверточной сети
            # обучим нейросеть с указанными параметрами

            optimizer = optim.Adam(cnn.parameters(), lr=0.0005, weight_decay=WD)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.03)
            criterion = nn.BCELoss()
            criterion.to(device)
            coach_cnn = TorchTrain(cnn, device=device, multigpu=False, metrics={'acc': helpers.acc_binary},
                                   model_type='cnn')
            coach_cnn.train(
                train_iterator,
                test_iterator,
                criterion,
                optimizer,
                n_epochs=N_EPOCHS,
                scheduler=scheduler,
                early_stoping=ES,
            )

            # сохраним обученную нейросеть
            self.models['cnn'].append(cnn)
            # посчитаем ответы сверточной нейросети на тестовом фолде
            test_ress_cnn = coach_cnn.test_res(test_iterator)

            print('Start lstm')
            # создадим модель lstm нейросети
            lstm = SentimentModel(
                torch.from_numpy(self.embed_matrix).float(),
                hidden_dim=256,
                output_dim=300,
                n_layers=2,
                bidirectional=True,
                dropout=0.4,
            )
            # переместим ее на заданный девайс
            lstm.to(device)

            # зададим алгоритм обучения lstm нейросети
            # зададим планировщик изменения параметра скорости обучения нейросети по экспоненциальному закону
            # выберем функцию ошибок - бинарную кросс-энтропию
            # создадим объект для обучения сверточной сети
            # обучим нейросеть с указанными параметрами

            optimizer = optim.Adam(lstm.parameters(), lr=0.0005, weight_decay=WD)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.03)
            criterion = nn.BCELoss()
            criterion.to(device)
            coach_lstm = TorchTrain(lstm, device=device, multigpu=False,
                                    metrics={'acc': helpers.acc_binary},
                                    model_type='lstm')
            coach_lstm.train(
                train_iterator,
                test_iterator,
                criterion,
                optimizer,
                n_epochs=N_EPOCHS,
                scheduler=scheduler,
                early_stoping=ES,
            )

            # сохраним обученную нейросеть
            self.models['lstm'].append(lstm)
            # посчитаем ответы lstm нейросети на тестовом фолде
            test_ress_lstm = coach_lstm.test_res(test_iterator)

            print('Start gru')
            # создадим модель gru нейросети
            gru = SentimentGRU(
                torch.from_numpy(self.embed_matrix).float(),
                hidden_dim=256,
                output_dim=300,
                n_layers=2,
                bidirectional=True,
                dropout=0.4,
            )
            # переместим ее на заданный девайс
            gru.to(device)

            # зададим алгоритм обучения gru нейросети
            # зададим планировщик изменения параметра скорости обучения нейросети по экспоненциальному закону
            # выберем функцию ошибок - бинарную кросс-энтропию
            # создадим объект для обучения gru сети
            # обучим нейросеть с указанными параметрами

            optimizer = optim.Adam(gru.parameters(), lr=0.0005, weight_decay=WD)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.03)
            criterion = nn.BCELoss()
            criterion.to(device)
            coach_gru = TorchTrain(gru, device=device, multigpu=False, metrics={'acc': helpers.acc_binary}, model_type='gru')
            coach_gru.train(train_iterator, test_iterator, criterion, optimizer, n_epochs=N_EPOCHS, scheduler=scheduler,
                            early_stoping=ES)

            # сохраним обученную нейросеть
            self.models['gru'].append(gru)
            # посчитаем ответы lstm нейросети на тестовом фолде
            test_ress_gru = coach_gru.test_res(test_iterator)


            # запишем ответы на данном фолде в матрицу out-of-fold ответов
            oof_train[idx_test, 0] = test_ress_cnn.cpu().numpy()
            oof_train[idx_test, 1] = test_ress_lstm.cpu().numpy()
            oof_train[idx_test, 2] = test_ress_gru.cpu().numpy()


        print('Start Stacking')
        # зададим вектор ответов out-of-fold на стекинге
        oof_stacking = np.zeros(oof_train.shape[0])

        # зададим структуру для сохранения обученных моделей стекинга
        self.stack_models = []
        # на 5 фолдах обучим модели стекинга
        for idx_train, idx_test in stratify_cv.split(df, df['answer']):
            # приготовим данные для обучения на данном фолде
            X_train = oof_train[idx_train]
            y_train = df['answer'].iloc[idx_train].values
            # приготовим данные для валидации на данном фолде
            X_test = oof_train[idx_test]
            y_test = df['answer'].iloc[idx_test].values

            # зададим модель градиентного бустинга
            model = LGBMClassifier()
            # обучим модель
            model.fit(X_train, y_train)
            # сохраним модель
            self.stack_models.append(model)
            # сделаем предсказания на валидации
            preds = model.predict_proba(X_test)[:, 1]
            # запишем ответы в вектор ответов out-of-fold на стекинге
            oof_stacking[idx_test] = preds

        # сохраним результаты как поля класса
        self.oof_stacking = oof_stacking
        self.oof_train = oof_train
        # найдем оптимальный порог для метрики accuracy
        self.th = helpers.max_th(df['answer'], oof_stacking)
        return oof_stacking

    def predict(self, df):
        # скопируем данные
        new_df = df.copy()
        if 'label' not in new_df:
            new_df['label'] = np.nan

        # установим параметры для предсказания
        BATCH_SIZE = 256 # размер батча
        device = self.device # девайс, на котором будут проводиться вычисления
        # создадим матрицу для out-of-fold предсказаний
        oof_test = np.zeros((new_df.shape[0], self.n_models))

        # в цикле по фолдам вычисляем предсказания
        for fold in range(self.n_folds):
            print('FOLD - {fold}'.format(fold=fold))
            # создаем объект для итерации по новым данным
            new = SimEmbDataset(new_df, self.TEXT)
            new_iterator = torch.utils.data.DataLoader(new, batch_size=BATCH_SIZE, shuffle=False,
                                                       num_workers=app_env.ml_params_data_loader_num_workers())

            print('Start cnn')
            # загрузим обученную сверточную нейросеть
            cnn = self.models['cnn'][fold]
            # переместим ее на девайс
            cnn.to(device)
            # создадим объект для предсказания
            coach_cnn = TorchTrain(cnn, device=device, multigpu=False,
                                   metrics={'acc': helpers.acc_binary}, model_type='cnn')
            # вычислим предсказания на новых данных
            test_new_cnn = coach_cnn.test_res(new_iterator)

            print('Start lstm')
            # загрузим обученную lstm нейросеть
            lstm = self.models['lstm'][fold]
            # переместим ее на девайс
            lstm.to(device)
            # создадим объект для предсказания
            coach_lstm = TorchTrain(lstm, device=device, multigpu=False,
                                    metrics={'acc': helpers.acc_binary}, model_type='lstm')
            # вычислим предсказания на новых данных
            test_new_lstm = coach_lstm.test_res(new_iterator)

            print('Start gru')
            # загрузим обученную gru нейросеть
            gru = self.models['gru'][fold]
            # переместим ее на девайс
            gru.to(device)
            # создадим объект для предсказания
            coach_gru = TorchTrain(gru, device=device, multigpu=False,
                                   metrics={'acc': helpers.acc_binary}, model_type='gru')
            # вычислим предсказания на новых данных
            test_new_gru = coach_gru.test_res(new_iterator)

            # записываем итоговые ответы в матрицу предсказаний
            oof_test[:, 0] += test_new_cnn.cpu().numpy() / self.n_folds
            oof_test[:, 1] += test_new_lstm.cpu().numpy() / self.n_folds
            oof_test[:, 2] += test_new_gru.cpu().numpy() / self.n_folds


        print('Start Stacking')
        # зададим вектор предсказаний
        new_preds_stacking = np.zeros(oof_test.shape[0])
        # в цикле получим предсказания стекинга моделей
        for model in self.stack_models:
            new = oof_test
            # получим предсказания модели
            preds_new = model.predict_proba(new)[:, 1]
            new_preds_stacking += preds_new / self.n_folds
        # возвращаем итоговые предсказания
        return new_preds_stacking

    def create_models_dir(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    def save_models_to_dir(self, path):
        for key in self.models:
            dir_path = os.path.join(path, 'models', key)
            self.create_models_dir(dir_path)
            for i, model in enumerate(self.models[key]):
                # torch.save(model.state_dict(), os.path.join(dir_path, "{i}.pt".format(i=i)))
                torch.save(model, os.path.join(dir_path, "{i}.pt".format(i=i)))

    def load_models_form_dir(self, path):
        for model_type in ['cnn', 'lstm', 'gru']:
            models_dir = os.path.join(path, 'models', model_type)
            for model_file_name in os.listdir(models_dir):    
                if self.models.get(model_type) is None:
                    self.models[model_type] = []
                model_path = os.path.join(models_dir, model_file_name)
                print(model_path)
                model = self.load_model(model_path, model_type)
                print(model)
                self.models[model_type].append(model)

    def load_model(self, path, model_type):
        model = torch.load(path, map_location = self.device)
        model.eval()
        return model

    def save(self, path):
        if os.path.isfile(path):
            joblib.dump((self.models, self.stack_models, self.oof_stacking, self.TEXT.vocab, self.th), path)
        else:
            self.create_models_dir(path)
            joblib_bump_path = os.path.join(path, 'other.pkl')
            joblib.dump((self.stack_models, self.oof_stacking, self.TEXT.vocab, self.th), path + '/other.pkl')
            self.save_models_to_dir(path)

    def load(self, path):
        if os.path.isfile(path):
            self.models, self.stack_models, self.oof_stacking, self.TEXT.vocab, self.th = joblib.load(path)
        else:
            self.models = {}
            self.stack_models, self.oof_stacking, self.TEXT.vocab, self.th = joblib.load(path + '/other.pkl')
            self.load_models_form_dir(path)