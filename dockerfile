FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    opencv-contrib-python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
