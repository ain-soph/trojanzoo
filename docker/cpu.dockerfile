FROM python:latest
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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir sphinx sphinxcontrib.katex numpy pyyaml pandas tqdm matplotlib seaborn scikit-learn && \
    pip install --no-cache-dir torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir trojanzoo
WORKDIR /trojanzoo/
