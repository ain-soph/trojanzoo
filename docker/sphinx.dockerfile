FROM python:latest
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y apt-utils && \
    apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt install -y make && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir sphinx sphinxcontrib.katex numpy pyyaml pandas tqdm matplotlib seaborn scikit-learn && \
    pip install --no-cache-dir torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /trojanzoo/

# docker build . --file ./sphinx.dockerfile  -t local0state/trojanzoo:sphinx
# docker push local0state/trojanzoo:sphinx