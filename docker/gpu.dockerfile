FROM local0state/base:gpu-conda
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"
LABEL org.opencontainers.image.source=https://github.com/ain-soph/trojanzoo

RUN cd / && \
    git clone https://github.com/ain-soph/trojanzoo.git && \
    cd ./trojanzoo && \
    pip install --no-cache-dir --upgrade -e .
WORKDIR /trojanzoo/
