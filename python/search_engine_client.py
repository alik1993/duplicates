import requests

import app_env


class SearchEngineClient:
    def __init__(self, endpoint=None):
        self.__endpoint = endpoint
        pass

    def get_ideas(self, content, limit=20):
        self.__valida_params()

        params = {"q": content, "limit": limit}

        response = requests.post(self.full_url('/search'), data=params)

        if response.status_code == 200:
            data = response.json()
            return data.get('idea_ids', [])
        else:
            return []

    def full_url(self, path):
        return "{endpoint}{path}".format(endpoint=self.endpoint(), path=path)

    def endpoint(self):
        if self.__endpoint is not None:
            return self.__endpoint
        else:
            return app_env.mp_params_scoring_search_engine_endpoint()

    def __valida_params(self):
        if self.endpoint() is None:
            raise Exception("Need set endpoint varibable or set env variable 'MP_PARAMS_SCORING_SEARCH_ENGINE_ENDPOINT'")

