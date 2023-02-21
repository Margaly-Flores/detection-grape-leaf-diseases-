FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

CMD ["python3", "main.py"]