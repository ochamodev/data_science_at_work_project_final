FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

COPY . /app

RUN pip install --upgrade pip --no-cache-dir -r requirements.txt

ENTRYPOINT ["python, batch_predict.py"]