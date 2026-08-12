FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY preprocessing ./preprocessing
COPY models ./models
COPY rag ./rag
COPY agents ./agents
COPY tools ./tools
COPY explainability ./explainability
COPY document_intelligence ./document_intelligence
EXPOSE 7860
CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","7860"]
