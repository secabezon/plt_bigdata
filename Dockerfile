FROM python:3.12-slim

COPY ./app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["mlflow","ui","--host","0.0.0.0","--port","5000"]