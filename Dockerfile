FROM ubuntu:18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y doxygen graphviz