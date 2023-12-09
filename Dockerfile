FROM ubuntu:22.04
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
# docker build -t cs61c-iamge-name .
# docker run -it --name your-container-name cs61c-iamge-name