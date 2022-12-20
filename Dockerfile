FROM --platform=linux/amd64 python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    python3-opencv \
    build-essential \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /code && mkdir /code/models && mkdir /code/models/rec && mkdir /code/models/det && mkdir /code/models/cls
WORKDIR /code

COPY . .

RUN python3.9 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    streamlit \
    paddlepaddle \
    paddleocr

WORKDIR /code/models/rec
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar \
    && tar xf en_PP-OCRv3_rec_infer.tar \
    && rm en_PP-OCRv3_rec_infer.tar

WORKDIR /code/models/det
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar \
    && tar xf en_PP-OCRv3_det_infer.tar \
    && rm en_PP-OCRv3_det_infer.tar

WORKDIR /code/models/cls
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar \
    && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar \
    && rm ch_ppocr_mobile_v2.0_cls_infer.tar
    
WORKDIR /code
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8080"]
