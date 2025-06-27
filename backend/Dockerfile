FROM python:3.10

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installe d'abord pip à jour
RUN pip install --no-cache-dir --upgrade pip --default-timeout=100 -i https://pypi.python.org/simple

# Installe torch/torchvision/torchaudio AVANT le reste (pour Detectron2)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --default-timeout=600 --retries 10 --progress-bar off \
    && pip install torchvision --index-url https://download.pytorch.org/whl/cpu --default-timeout=600 --retries 10 --progress-bar off \
    && pip install torchaudio --index-url https://download.pytorch.org/whl/cpu --default-timeout=600 --retries 10 --progress-bar off

# Installe le reste des dépendances (dont detectron2)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt -i https://pypi.python.org/simple

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]