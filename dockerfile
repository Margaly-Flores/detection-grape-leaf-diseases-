FROM python:3.8.13

RUN apt-get update && apt-get install -y \
    opencv-contrib-python 
COPY requirements.txt requirements.txt