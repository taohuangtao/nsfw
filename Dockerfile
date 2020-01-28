FROM ubuntu:latest
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y python3 --fix-missing && apt-get install -y python3-pip --fix-missing
RUN mkdir -p /root/.pip/
COPY pip.conf /root/.pip/pip.conf
RUN pip3 install tensorflow==1.9 && pip3 install numpy==1.16.0 && pip3 install opencv-python \
    && pip3 install matplotlib && pip3 install Pillow \
    && rm -rf /root/.cache/pip
RUN pip3 install Flask && pip3 install nltk && pip3 install jieba && pip3 install PyMySQL && pip3 install sqlalchemy \
    && pip3 install flask_sqlalchemy && pip3 install flask_jsontools && pip3 install redis && pip3 install gevent \
    && rm -rf /root/.cache/pip
WORKDIR /workspace
COPY data /workspace/data
COPY eagle /workspace/eagle
COPY resnet /workspace/resnet
COPY route /workspace/route
COPY utils /workspace/utils
COPY config.py /workspace
COPY nsfw_predict.py /workspace
COPY nsfw_predict_api.py /workspace
COPY rest_api.py /workspace
COPY serving_client.py /workspace
COPY start_tensorflow_serving.sh /workspace

CMD ["python3", "./rest_api.py"]
EXPOSE 5000