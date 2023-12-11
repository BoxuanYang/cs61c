FROM ubuntu:18.04
COPY . /cs61c/
RUN apt-get update && apt-get install -y \
    git \
    python3 \
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

