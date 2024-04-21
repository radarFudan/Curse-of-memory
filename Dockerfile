FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

USER root:root

ARG DEBIAN_FRONTEND=noninteractive

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Common Dependencies
RUN apt-get update && \
    # Others
    apt-get install -y libksba8 \
    openssl \
    libaio-dev \
    git \
    wget

# Install dependencies
COPY apt_install.txt .
RUN apt-get update
RUN apt-get install -y `cat apt_install.txt`
RUN rm apt_install.txt

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip, install py libs
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt --upgrade
RUN rm requirements.txt

RUN git clone https://github.com/radarFudan/Curse-of-memory.git
