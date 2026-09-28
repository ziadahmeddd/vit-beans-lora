FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 MODEL_DIR=/srv/models/vit-beans-int8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY app/ .
COPY models/vit-beans-int8/ models/vit-beans-int8/

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
