FROM railwayapp/base:latest

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    opencv-contrib-python

RUN find / -name "libGL.so.1" 2>/dev/null

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
