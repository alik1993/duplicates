import os

import sqlalchemy


def data_model():
    return os.getenv('MODEL_DATA', os.path.join('..', 'data'))


def data_model_builtin():
    return os.path.join(data_model(), 'builtin')


def data_model_builtin_path(path):
    return os.path.join(data_model_builtin(), path)


def data_model_runtime():
    return os.path.join(data_model(), 'runtime')


def data_model_runtime_path(path, not_builtin=False):
    rt_path = os.path.join(data_model_runtime(), path)
    if not_builtin is True:
        return rt_path

    if os.path.exists(rt_path):
        return rt_path
    else:
        return data_model_builtin_path(path)


def database_url():
    return os.environ['DATABASE_URL']


def database_search_path():
    return os.getenv('DATABASE_SEARCH_PATH', 'tenant1,public,extensions')


def data_model_learn():
    return os.path.join(data_model_builtin(), 'learn')


def data_model_learn_path(path):
    return os.path.join(data_model_learn(), path)


def data_model_learn_duplicats_log():
    return data_model_learn_path('duplicates.log')


def db_engine():
    return sqlalchemy.create_engine(database_url(),
                                    connect_args={'options': '-csearch_path={}'.format(database_search_path())})


def ml_params_use_gpu():
    return os.getenv('ML_PARAMS_USE_GPU', 'false') == 'true'


def ml_params_cuda_devise_id():
    return os.getenv('ML_PARAMS_CUDA_DEVISE_ID', '')


def ml_params_cuda_devise():
    res = ['cuda']
    if ml_params_cuda_devise_id() != ():
        res.append(ml_params_cuda_devise_id())

    return ':'.join(res)


def ml_params_vcpu_limit():
    return int(os.getenv('ML_PARAMS_VCPU_LIMIT', '2'))


def ml_params_memory_limit():
    return int(os.getenv('ML_PARAMS_MEMORY_LIMIT', '1024'))


def ml_params_data_loader_num_workers():
    return int(os.getenv('ML_PARAMS_DATA_LOADER_NUM_WORKERS', '4'))


def ml_params_dataset_learn_limit():
    v = os.getenv('ML_PARAMS_DATASET_LEARN_LIMIT', None)
    if v is None:
        return 0
    else:
        return int(v)


def ml_params_dataset_scoring_limit():
    v = os.getenv('ML_PARAMS_DATASET_SCORING_LIMIT', None)
    if v is None:
        return None
    else:
        return int(v)


def ml_params_scoring_search_filter():
    return os.getenv('MP_PARAMS_SCORING_SEARCH_FILTER', 'search_engine')


def mp_params_scoring_search_engine_endpoint():
    return os.getenv('MP_PARAMS_SCORING_SEARCH_ENGINE_ENDPOINT')
