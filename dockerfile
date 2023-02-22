# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=1
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y \
    python-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install libgl1-mesa-glx

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run app.py when the container launches
CMD ["python", "main.py"]