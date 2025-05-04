FROM python:3.11-slim

# system libs for xgboost wheel
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    ALPHA_VANTAGE_API_KEY=BN3UF29VKSR67YHN \
    ALPACA_API_KEY=PKB3GME4Y837WW3HQS00 \
    ALPACA_END_POINT=https://paper-api.alpaca.markets/ \
    ALPACA_SECRET_KEY=TuFcUldn4HmBjPocR7cUH9SudJQvlCQwf0uXvKxg

EXPOSE 8501
CMD ["streamlit", "run", "predictions.py"]
