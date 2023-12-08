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