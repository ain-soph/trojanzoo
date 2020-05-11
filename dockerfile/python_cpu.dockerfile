FROM ubuntu:latest
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"
RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y \
    tzdata \
    wget \
    apt-utils \
    build-essential \
    cmake \
    golang \
    default-jre \
    default-jdk \
    git \
    libopencv-dev \
    libsnappy-dev \
    vim \
    tmux \
    zip \
    axel && \
    apt upgrade -y && \
    apt clean && \
    ln -sf /usr/share/zoneinfo/EST /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \ 
    sh -c '/bin/echo -e "\nyes\n\nyes" | sh ~/anaconda.sh' && \
    rm ~/anaconda.sh && \
    /root/anaconda3/bin/conda update --all && \
    /root/anaconda3/bin/pip install --upgrade pip && \
    /root/anaconda3/bin/pip install opencv-python && \
    /root/anaconda3/bin/conda install -y autopep8 pylint && \
    conda install pytorch torchvision cpuonly -c pytorch && \
    /root/anaconda3/bin/pip install tensorflow
    
WORKDIR /root/
# Install basic dependencies
# Install anaconda
# Set timezone
# Initialize workspace
# /root/anaconda3/bin/conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch && \
# /root/anaconda3/bin/conda install cudnn=7.6 cudatoolkit=10.0 tensorflow-gpu
# curl --silent https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -o ~/anaconda.sh
# # Set timezone
# RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# # Set locale
# ENV LANG C.UTF-8 LC_ALL=C.UTF-8
    # /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-scatter && \
    # /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-sparse && \
    # /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-cluster && \
    # /root/anaconda3/bin/pip install --verbose --no-cache-dir torch-spline-conv && \
    # /root/anaconda3/bin/pip install torch-geometric