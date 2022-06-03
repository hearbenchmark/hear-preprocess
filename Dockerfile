# Unfortunately, dockerhub won't integrate with github unless we
# pay, so just run this manually each time you update the Dockerfile:
# docker build -t turian/hearpreprocess . && docker push turian/hearpreprocess

FROM ubuntu

ENV LANG C.UTF-8

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update
    #   apt-get update && \

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        less \
        locate

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        unzip unrar \
        zip \
        tmux \
        screen \
        bc \
        openssh-client

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && apt-get install -y -q

# ==================================================================
# python
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        python3-pip

#        && \
#    add-apt-repository ppa:deadsnakes/ppa && \
#    apt-get update && \
#    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
#        python3.7 \
#        python3.7-dev \
#        python3-distutils-extra \
#        && \
#    wget -O ~/get-pip.py \
#        https://bootstrap.pypa.io/get-pip.py && \
#    python3.7 ~/get-pip.py && \
#    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
#    ln -s /usr/bin/python3.7 /usr/local/bin/python

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        setuptools \
        pipdeptree \
        ipython

# ==================================================================
# hearpreprocess
# ------------------------------------------------------------------

RUN apt update
RUN apt upgrade -y
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL \
        software-properties-common \
        sox \
        ffmpeg \
        libvorbis-dev \
        vorbis-tools

## For ffmpeg >= 4.2
## Could also build from source:
## https://github.com/jrottenberg/ffmpeg/blob/master/docker-images/4.3/ubuntu1804/Dockerfile
#RUN add-apt-repository ppa:jonathonf/ffmpeg-4
#RUN apt-get update
#RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
#    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
#    GIT_CLONE="git clone --depth 10" && \
#    $APT_INSTALL ffmpeg

# gsutil
# https://cloud.google.com/storage/docs/gsutil_install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL apt-transport-https ca-certificates gnupg
RUN wget -O - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    $APT_INSTALL google-cloud-sdk
#gcloud init

RUN echo 20211120

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install" && \
    GIT_CLONE="git clone --depth 10" && \
    $GIT_CLONE https://github.com/hearbenchmark/hear-preprocess.git
RUN cd hear-preprocess && \
    python3 -m pip --no-cache-dir install -e ".[dev]"

RUN updatedb

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006
