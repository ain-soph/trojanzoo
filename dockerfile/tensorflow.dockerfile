FROM nvidia/cuda:10.0-devel
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
    /root/anaconda3/bin/conda install cudnn=7.6 cudatoolkit=10.0 tensorflow-gpu
WORKDIR /root/