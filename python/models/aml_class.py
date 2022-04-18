import sber_ailab_automl as saa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='[%(asctime)-15s] %(levelname)-8s: %(message)s')
logger = logging.getLogger(__name__)

import app_env

class AMLSolver():
    def __init__(self, path='dat'):
        params = {}
        params['vCPULimit'] = app_env.ml_params_vcpu_limit()
        params['memoryLimit'] = app_env.ml_params_memory_limit()
        params['timeLimit'] = 300 * 600
        params['type'] = 'classification'
        params['trainDatasetFName'] = path + '/train.csv'
        params['testDatasetFName'] = path + '/holdout.csv'
        params['modelsPath'] = path + '/models1'
        params['predictionsFName'] = path + '/predictions/preds.csv'
        params['encoding'] = 'utf-8'
        params['separator'] = r","
        params['decimal'] = '.'
        params['debugFlag'] = True
        params['featureRoles'] = {
            'target': 'label',
            'line_id': 'proj_id',
            'drop': [],
            'numeric': [],
            'datetime': [],
            'string': [],
            'id': [],
            'text': ['text1', 'text2']
        }
        params['naValues'] = None
        params['rowsToAnalyze'] = 1000
        params['datetimeFormat'] = '%Y-%m-%d'
        params['embeddingsPath'] = app_env.data_model_builtin_path('bank_fasttextskipgram_300')
        params['useGPU'] = app_env.ml_params_use_gpu()
        params['testBatchSize'] = 500000
        params['LuckySeed'] = 777
        params['KFolds'] = 10
        
        self.params = params
        
        self.automl = saa.AutoML(
#             model_dir = self.params['modelsPath'], 
            vCPULimit = self.params['vCPULimit'], 
            memoryLimit = self.params['memoryLimit'], 
            timeLimit = self.params['timeLimit'], 
            encoding = self.params['encoding'], 
            separator = self.params['separator'], 
            decimal = self.params['decimal'], 
            datetime_format = self.params['datetimeFormat'], 
            na_values = self.params['naValues'], 
            analyze_rows = self.params['rowsToAnalyze'],
            embeddings_path = self.params['embeddingsPath'],
            use_gpu = self.params['useGPU'],
            test_batch_size = self.params['testBatchSize'],
            cv_random_state = self.params['LuckySeed'],
            KFolds = self.params['KFolds']
        )

            
        self.train_oof = None
        self.test_pred = None
        
    def train(self, train_file_path):
        self.automl = saa.AutoML(
#             model_dir = self.params['modelsPath'], 
            vCPULimit = self.params['vCPULimit'], 
            memoryLimit = self.params['memoryLimit'], 
            timeLimit = self.params['timeLimit'], 
            encoding = self.params['encoding'], 
            separator = self.params['separator'], 
            decimal = self.params['decimal'], 
            datetime_format = self.params['datetimeFormat'], 
            na_values = self.params['naValues'], 
            analyze_rows = self.params['rowsToAnalyze'],
            embeddings_path = self.params['embeddingsPath'],
            use_gpu = self.params['useGPU'],
            test_batch_size = self.params['testBatchSize'],
            cv_random_state = self.params['LuckySeed'],
            KFolds = self.params['KFolds']
        )
        
        res_df_oof, feat_imp = self.automl.train(train_file_path, 
                                                 self.params['featureRoles'], 
                                                 self.params['type'], 
                                                 use_ids = False)
        self.automl.save(self.params['modelsPath'] + '/tmp.model_automl')
        self.train_oof = res_df_oof[['REG', 'GBM']]
        return res_df_oof[['REG', 'GBM']]
    
    def test(self, test_file_path):
        self.automl = saa.AutoML(
#             model_dir = self.params['modelsPath'], 
            vCPULimit = self.params['vCPULimit'], 
            memoryLimit = self.params['memoryLimit'], 
            timeLimit = self.params['timeLimit'], 
            encoding = self.params['encoding'], 
            separator = self.params['separator'], 
            decimal = self.params['decimal'], 
            datetime_format = self.params['datetimeFormat'], 
            na_values = self.params['naValues'], 
            analyze_rows = self.params['rowsToAnalyze'],
            embeddings_path = self.params['embeddingsPath'],
            use_gpu = self.params['useGPU'],
            test_batch_size = self.params['testBatchSize'],
            cv_random_state = self.params['LuckySeed'],
            KFolds = self.params['KFolds']
        )
        self.automl.load(app_env.data_model_runtime_path('model1/tmp.model_automl'))
        
        preds_df, score, res_df = self.automl.predict(test_file_path, 
                                                      self.params['predictionsFName'])
        preds_df.columns = ['reg', 'gbm']
        self.test_pred = preds_df[['reg', 'gbm']]
        self.test_pred.columns = ['REG', 'GBM']
        return self.test_pred
    
    # def _get_test_score(self):
    #     def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    #         df = pd.merge(preds, pd.read_csv(target_csv), right_on="line_id", left_index=True)
    #         score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
    #             mean_squared_error(df.target.values, df.prediction.values) ** 0.5
    #
    #         return score
    #
    #     if score is None:
    #         score = validate(pd.read_csv(self.params['predictionsFName'], index_col = 'line_id'),
    #                          self.params['testDatasetFName'].replace('test', 'test-target'),
    #                          self.params['type'])
    #     logging.info(f'Score is {score}')