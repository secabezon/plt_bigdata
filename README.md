# plt_bigdata

Repositorio generado para proyecto plataforma BigData 


## CON DOCKER 

### Para Generar imagen:

docker buildx build -t apppltfrm .

### Para Levantar contenedor

docker run -d -p 5000:5000 --name aplicacion_mlflow apppltfrm mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow.db --workers 1

## SIN DOCKER 

### Para levantar MLFLOW

mlflow ui --backend-store-uri sqlite:///mlflow.db


## Entreno modelo con comando 

python .\app\execute_model_mlflow.py

python .\app\generate_batch.py

## MODEL SERVE

$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

mlflow models serve -m models:/model_prod/2 --no-conda -p 4000

mlflow models predict -m "models:/model_prod/2" -i app/data/batch_pred_procesado.csv -o app/data/predict.csv --env-manager local --content-type csv