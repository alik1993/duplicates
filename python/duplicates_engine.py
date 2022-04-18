import numpy as np

from duplicates_model import DuplicatesModels
from learn_engine import LearnEngine

import app_env
import helpers
from data_loader import DataLoader

import os

class DuplicatesEngine:
    def __init__(self):
        self.md = DuplicatesModels()
        self.md_loaded = False

    def get_idea_duplicates(self, idea_id, with_own_project=True, project_ids=None,
                            idea_batch_size=app_env.ml_params_dataset_scoring_limit()):
        if project_ids is None:
            project_ids = []

        dl = DataLoader()
        df = dl.ideas(idea_id, with_own_project=with_own_project, project_ids=project_ids)
        df['answer'] = np.nan

        result = self.idea_duplicates_by_batch(idea_id, df, idea_batch_size=idea_batch_size)

        return result

    def idea_duplicates_by_batch(self, idea_id, df,
                                 idea_batch_size=app_env.ml_params_dataset_scoring_limit()):
        result = {}

        if idea_batch_size is not None:
            index_start = 0
            index_stop = idea_batch_size
            while True:
                df1 = df.iloc[index_start:index_stop]
                if len(df1) == 0:
                    break
                result_batch = self.__idea_duplicates(idea_id, df1)
                for k in result_batch:
                    v = result.setdefault(k, [])
                    v += result_batch[k]
                index_start = index_stop
                index_stop += idea_batch_size
        else:
            result = self.__idea_duplicates(idea_id, df)

        return result

    def __idea_duplicates(self, idea_id, df):
        self.load_md()

        results_proba = self.md.predict(df)
        results = helpers.expand(df).copy()
        results['proba'] = list(results_proba) + list(results_proba)
        results['is_duplicate'] = (results['proba'] > self.md.th).astype(int)
        results['scoring_search_filter'] = df['scoring_search_filter'].values[0]

        def project_apply(row):
            if row.is_own_project is True:
                return 'own_project'
            else:
                return row.project_id2

        results['project'] = results[['project_id2', 'is_own_project']].apply(project_apply, axis=1)

        results_dups = results[results['is_duplicate'] == 1]
        results_dups.sort_values(by='proba', ascending=False, inplace=True)

        result_ideas = results_dups[results['id2'] != idea_id]
        result_ideas = result_ideas[['id2', 'proba', 'project']].copy()
        result_ideas = result_ideas.rename(columns={'id2': 'id', 'proba': 'score'})

        result_array = result_ideas.to_dict('records')
        result_dict = {}
        for row in result_array:
            v = result_dict.setdefault(row['project'], [])
            row_copy = row.copy()
            del row_copy['project']
            v.append(row_copy)

        return result_dict

    def ideas_for_project(self, project_id):
        return DataLoader().ideas_in_project(project_id)

    def learn(self):
        learn_engine = LearnEngine(self.md)

        self.md = learn_engine.learn()
        self.md_loaded = True

    def load_md(self):
        if self.md_loaded is False:
            data_dir = app_env.data_model_runtime_path('v3')
            if(os.path.isdir(data_dir)):
                path_for_load = data_dir
            data_file = app_env.data_model_runtime_path('v3.pkl')
            if(os.path.isfile(data_file)):
                path_for_load = data_file
            self.md.load(path_for_load)
            self.md_loaded = True
