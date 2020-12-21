FROM python:latest
LABEL maintainer="Ren Pang <rbp5354@psu.edu>"
RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y apt-utils
    apt upgrade -y && \
RUN DEBIAN_FRONTEND="noninteractive" apt install -y make && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install numpy sphinx sphinxcontrib.katex pyyaml pandas tqdm matplotlib seaborn scikit-learn && \
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # pip install https://files.pythonhosted.org/packages/41/f4/4da4f26a04d93851e481e76ec17fed0d152a1691e8f1142ad763c9f07997/torch-1.7.1-cp39-cp39-manylinux1_x86_64.whl && \
    # pip install https://files.pythonhosted.org/packages/18/39/64aa235e1f08c27d9fafd3aa317d081b45914f68a7a2a87b488fd0b69589/torchvision-0.8.2-cp39-cp39-manylinux1_x86_64.whl
WORKDIR /trojanzoo/
