FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"
# Install softwares
RUN apt update && \
    # Install basic dependencies
    DEBIAN_FRONTEND="noninteractive" apt install -y \
    tzdata \
    wget \
    apt-utils \
    git \
    vim \
    tmux && \
    # build-essential \
    # cmake \
    # golang \
    # default-jre \
    # default-jdk \
    # libopencv-dev \
    # libsnappy-dev \
    # zip \
    # axel \
    apt upgrade -y && \
    apt clean && \
    # Set timezone
    ln -sf /usr/share/zoneinfo/EST /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata
# Install python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \ 
    sh -c '/bin/echo -e "\nyes\n\nyes" | sh ~/anaconda.sh' && \
    rm ~/anaconda.sh && \
    /root/anaconda3/bin/conda update --all && \
    /root/anaconda3/bin/pip install --upgrade pip && \
    /root/anaconda3/bin/conda install -y pytorch torchvision cudatoolkit=11.0 -c pytorch
    # /root/anaconda3/bin/pip install opencv-python && \
    # /root/anaconda3/bin/conda install -y autopep8 pylint && \
# Install trojanzoo
WORKDIR /root/
# /root/anaconda3/bin/pip install tensorflow && \
# curl --silent https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -o ~/anaconda.sh && \

# RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# Set locale
# ENV LANG C.UTF-8 LC_ALL=C.UTF-8

# /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-scatter && \
# /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-sparse && \
# /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-cluster && \
# /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-spline-conv && \
# /root/anaconda3/bin/pip install torch-geometric