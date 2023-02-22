# Use an official Python runtime as a parent image
#FROM python:3.9-slim-buster
FROM ubuntu:latest

ENV PYTHONUNBUFFERED=1

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    python-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Set the working directory to /app
#WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app
# Run app.py when the container launches
CMD ["python", "main.py"]