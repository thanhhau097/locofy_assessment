FROM nvcr.io/nvidia/tensorrt:21.11-py3 as base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt -y upgrade && \
    apt install -y ffmpeg libsm6 libxext6 enchant myspell-es python3-pip build-essential && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN apt install graphviz -y

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
ADD . /workspace

ENTRYPOINT ["python", "main.py"]
