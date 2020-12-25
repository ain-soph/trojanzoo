FROM local0state/base:gpu-conda
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"

RUN pip install --no-cache-dir trojanzoo && \
    cd / && \
    git clone https://github.com/ain-soph/trojanzoo.git
WORKDIR /trojanzoo/
