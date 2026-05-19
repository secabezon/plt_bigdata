FROM python:3.12-slim

COPY ./app /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "execute_model_mlflow.py"]