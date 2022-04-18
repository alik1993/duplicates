import sqlalchemy
import pandas as pd
import json

import networkx as nx

import app_env
from duplicates_model import DuplicatesModels


class LearnEngine():
    def __init__(self, duplicates_model):
        self.duplicates_model = duplicates_model
        self._duplicates_log = None
        self._duplicates_data = None
        self._idea_ids_from_duplicates = None
        self._graphs_components = None

    def learn(self):
        dups = self.duplicates_data().copy()

        # проставим флаг: входит ли id первого текста в разметке в самую большую компоненту графа или нет
        dups['flag'] = dups['id1'].isin(self.graphs_components()).astype(int)

        # оставим данные для обучения те, в которых флаг равен 1
        df = dups[dups['flag'] == 1].reset_index(drop=True).copy()
        # оставим данные для отложенной выборки те, в которых флаг равен 0
        holdout = dups[dups['flag'] == 0].reset_index(drop=True).copy()

        if app_env.ml_params_dataset_learn_limit() != 0:
            print("Use learn dataset limit by {}".format(app_env.ml_params_dataset_learn_limit()))
            df = df.head(app_env.ml_params_dataset_learn_limit())

        print("Learn dataset count: {}".format(len(df)))

        duplicates_model = DuplicatesModels()

        duplicates_model.train(df)

        duplicates_model.save(app_env.data_model_runtime_path('v3.pkl', not_builtin=True))

        return duplicates_model

    def duplicates_log(self):
        if self._duplicates_log is not None:
            return self._duplicates_log

        # загрузим разметку по дубликатам
        log = []
        with open(app_env.data_model_learn_duplicats_log()) as f:
            log = [json.loads(x) for x in f.readlines()]

        # создадим pandas dataframe
        df_log = pd.DataFrame(log)
        # переведем ответы из разметки в числа 0 и 1
        df_log['answer'] = df_log['answer'].map({'yes': 1, 'no': 0})

        # посчитаем средний ответ и количество ответов по парам текстов (возможным дубликатам)
        df_log = df_log.groupby(by=['id1', 'id2'])['answer'].agg(['mean', 'count']).reset_index()
        # оставим только записи, которые были размечены ровно 3 раза
        df_log = df_log[df_log['count'] == 3]
        # правильными ответами будем считать 0 (не дубликаты), если усредненный по 3 эспертам ответ был 0,
        # и 1 (дубликаты), если усредненный по 3 эспертам ответ был больше 0
        df_log['answer'] = (df_log['mean'] > 0.0).astype(int)
        # так же уберем данные, которые только один эксперт ответил, что тексты в нем дубликаты
        df_log = df_log[(df_log['mean'] < 0.1) | (df_log['mean'] > 0.5)]

        self._duplicates_log = df_log

        return self._duplicates_log

    def duplicates_data(self):
        if self._duplicates_data is not None:
            return self._duplicates_data

        df_log = self.duplicates_log()

        df = self.ideas()
        # добавим к отфильтрованным данным тексты идей
        dups = df_log.join(df[['content', 'id']].set_index('id'), on='id1', rsuffix='1')
        dups = dups.join(df[['content', 'id']].set_index('id'), on='id2', rsuffix='2')
        dups = dups.rename({'content': 'content1'}, axis=1)

        self._duplicates_data = dups

        return self._duplicates_data

    def ideas(self):
        sql = """
SELECT id, content 
FROM pages 
WHERE type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')  
      AND id = ANY(:ids)
        """

        sql = sqlalchemy.text(sql)

        ideas = pd.read_sql_query(sql, app_env.db_engine(), params={'ids': list(self.idea_ids_from_duplicates())})

        return ideas

    def idea_ids_from_duplicates(self):
        if self._idea_ids_from_duplicates is not None:
            return self._idea_ids_from_duplicates

        ids = set(self.duplicates_log()['id1'])
        ids.update(self.duplicates_log()['id2'])

        self._idea_ids_from_duplicates = ids

        return self._idea_ids_from_duplicates

    def graphs_components(self):
        if self._graphs_components is not None:
            return self._graphs_components

        # для разбиения на обучающую и отложенную выборку создадим граф,
        # в котором вершинами являются id идей, а ребра ставятся между двумя вершинами если
        # между соответствующими id есть разметка (дубликат или не дубликат).
        # Это сделано для того, чтобы в отложенной выборке не было текстов из обучающей выборки

        # создаем граф
        G = nx.Graph()

        # добавляем вершины
        nodes = set(list(self.duplicates_data()['id1']) + list(self.duplicates_data()['id2']))
        G.add_nodes_from(nodes)

        # добавляем ребра
        edges = zip(list(self.duplicates_data()['id1']), list(self.duplicates_data()['id2']))
        G.add_edges_from(edges)

        # для разделения на обучающую и отложенную выборку найдем клики графа
        components = nx.connected_components(G)
        components = list(components)

        # выберем самую большую клику
        for comp in components:
            break

        self._graphs_components = comp

        return comp
