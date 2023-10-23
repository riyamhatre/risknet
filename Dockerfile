FROM ubuntu:18.04

ENV PYTHON_VERSION=3.9 

# change shell to bash which supports parameter expansion
SHELL ["/bin/bash", "-c"]

# install some essential utilities
RUN apt-get update && apt-get install curl -y && apt-get install vim -y && \
    apt-get install htop -y && apt-get install yarn -y && \
    curl -sL install-node.now.sh/lts | bash -s -- --yes

# install python and pip
RUN apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \
    export DEBIAN_FRONTEND="noninteractive" && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION:0:1}-pip && \
    apt-get install -y python${PYTHON_VERSION}-distutils && \
    apt-get install -y python${PYTHON_VERSION}-venv && \
    python${PYTHON_VERSION} -m pip install --upgrade setuptools && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    pip${PYTHON_VERSION} install build


###############################################################################

ADD . /

RUN make clean-install

RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]

