FROM python:3.10-slim

WORKDIR /app

# Instala gcc y python3-dev para las librerías que requieren compilación
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

COPY requirements.txt /app/
RUN pip install --upgrade pip --no-cache-dir -r requirements.txt

COPY kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

COPY . /app

# Descarga el dataset de Kaggle
RUN kaggle datasets download -d mlg-ulb/creditcardfraud -p /app/data --unzip

ENTRYPOINT ["python", "batch_predict.py", "--data_path", "/app/data/creditcard.csv"]
