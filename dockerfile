FROM python:3.8.13

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    opencv-contrib-python
COPY requirements.txt requirements.txt