ARG BASE_IMAGE_NAME=registry.daturum.ru/sberbank/agiki/data-models/base_images/base
ARG BASE_IMAGE_VERSION=latest
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_VERSION}

RUN mkdir /app

COPY . /app

WORKDIR /app

ENV MODEL_DATA /app/data
ENV ML_PARAMS_USE_GPU false
ENV DATABASE_SEARCH_PATH tenant1,public,extensions
ENV SERVER_PORT 9292
ENV SERVER_HOST 0.0.0.0

RUN cd /app/ && \
    bundle install && \
    cd vendor/python/automl_ai_lab && \
    python setup.py install --user && \
    cd /app && \
    pip3 install -r python/requirements.txt && \
    python -c "import nltk; nltk.download('stopwords')" && \
    rm -rf ~/nltk_data/corpora/stopwords.zip && \
    rm -rf ~/.cache/pip && \
    cd /app

CMD ["/app/bin/server"]
