FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODEL_DIR=/app/models DATA_DIR=/app/data TORCH_THREADS=2
WORKDIR /app
COPY service/requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY service/ service/
COPY models/ticket-clf models/ticket-clf
COPY models/ner-hybrid models/ner-hybrid
COPY models/minilm models/minilm
COPY data/kb.json data/kb.json
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
