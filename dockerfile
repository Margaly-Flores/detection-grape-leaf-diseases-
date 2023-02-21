FROM python:3
RUN pip install opencv-contrib-python
RUN apt-get update && apt-get install libgl1-mesa-glx
COPY requirements.txt requirements.txt