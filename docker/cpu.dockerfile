FROM local0state/base:cpu-conda
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"

RUN cd / && \
    git clone https://github.com/ain-soph/trojanzoo.git && \
    cd ./trojanzoo && \
    pip install --no-cache-dir --upgrade -e .
WORKDIR /trojanzoo/
