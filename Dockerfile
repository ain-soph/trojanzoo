FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y apt-utils && \
    apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt install -y vim tmux git make tzdata && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Set timezone
    ln -sf /usr/share/zoneinfo/EST /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.9 python3-pip
RUN pip install --upgrade pip && \
    pip install numpy sphinx sphinxcontrib.katex pyyaml pandas tqdm matplotlib seaborn scikit-learn && \
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/ain-soph/trojanzoo.git && \
    pip install -r ./trojanzoo/requirements.txt && \
    pip install -r ./trojanzoo/docs/requirements.txt
WORKDIR /trojanzoo/

# RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \ 
#     sh -c '/bin/echo -e "\nyes\n\nyes" | sh ~/anaconda.sh' && \
#     rm ~/anaconda.sh && \
#     /root/anaconda3/bin/conda update --all && \
#     /root/anaconda3/bin/pip install --upgrade pip && \
#     /root/anaconda3/bin/conda install -y pytorch torchvision cudatoolkit=11.0 -c pytorch
#     # /root/anaconda3/bin/pip install opencv-python && \
#     # /root/anaconda3/bin/conda install -y autopep8 pylint && \
# Install trojanzoo
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