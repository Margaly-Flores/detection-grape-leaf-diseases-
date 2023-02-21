FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python3", "main.py"]