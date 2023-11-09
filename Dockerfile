FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
USER root:root

ARG DEBIAN_FRONTEND=noninteractive

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

