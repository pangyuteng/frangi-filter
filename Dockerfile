#FROM tensorflow/tensorflow:2.6.1-gpu-jupyter
FROM ubuntu:18.04

RUN apt-get update && apt-get upgrade -y

# install MedPy depenencies
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip

# install graph-cut functionality dependencies
RUN apt-get install -y libboost-python-dev build-essential

COPY requirements.txt .
RUN pip install -r requirements.txt


