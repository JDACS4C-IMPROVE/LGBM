FROM python:3.7

RUN apt update -y
RUN apt install git

ENV IMPROVE_MODEL_DIR /usr/local/LGBM
ENV PYTHONPATH $PYTHONPATH:/IMPROVE/


# IMPROVE
RUN pip install git+https://github.com/ECP-CANDLE/candle_lib@develop # CANDLE
RUN git clone https://github.com/JDACS4C-IMPROVE/IMPROVE.git


# Model
RUN pip install pyarrow==12.0.1 # saves and loads parquet files
RUN pip install lightgbm==3.1.1

COPY . /usr/local/LGBM/
RUN cp /usr/local/LGBM/*.sh /usr/local/bin
# RUN chmod a+x /usr/local/bin/infer.sh /usr/local/bin/train.sh /usr/local/bin/preprocess.sh