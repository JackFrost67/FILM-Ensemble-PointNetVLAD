FROM nvidia/cuda:10.2-devel-ubuntu18.04
ARG TORCH_CUDA_ARCH_LIST="6.1+PTX"

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -yy \
  build-essential \
  zlib1g-dev \ 
  libncurses5-dev \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  libbz2-dev \
  liblzma-dev \
  libopenblas-dev \
  wget

# INSTALL PYTHON-3.8.15
RUN wget https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz
RUN tar xzf Python-3.8.15.tgz
WORKDIR Python-3.8.15

RUN ./configure
RUN make
RUN make install

RUN apt-get update && apt-get install -yy \
  git \
  software-properties-common \
  python3-pip \
  nano \
  && rm -rf /var/lib/apt/lists/*

COPY . /PointNetVlad-Pytorch

ENV PYTHONPATH "${PYTHONPATH}:/PointNetVlad-Pytorch" 

WORKDIR /PointNetVlad-Pytorch

RUN pip3 install -r requirements.txt

RUN git config --global --add safe.directory '*'

#ARG CUDA_VISIBLE_DEVICES=0

# ENTRYPOINT ["python", "datasets/pointnetvlad/generate_training_tuples_baseline.py", "--dataset", "datasets/benchmark_datasets"]