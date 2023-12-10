FROM ubuntu:22.04
COPY . /cs61c/
RUN apt-get update && apt-get install -y \
    libcunit1 libcunit1-doc libcunit1-dev
    git \
    pip \
    python3.10 \
    python3.10-venv \
    python3-dev \
    openjdk-17-jdk \
    gcc \
    gdb \
    g++ \
    make \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /cs61c 


# 运行这个docker，只需要执行如下命令：
#docker build -t cs61c-image .
#docker run -v C:/users/kevin/projects/cs61c:/cs61c -it cs61c-image

