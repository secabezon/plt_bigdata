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

## PREDICT BATCH

mlflow models predict -m "models:/model_prod/2" -i app/data/batch_pred_procesado.csv -o app/data/predict.csv --env-manager local --content-type csv

## PREDICT API

Abrir terminal BASH:

curl -X POST "http://127.0.0.1:4000/invocations" -H "Content-Type: application/json" -d '{"dataframe_split": {"columns": ["year","running","status","motor_volume","model_hyundai","model_kia","model_mercedes-benz","model_nissan","model_toyota","motor_type_diesel","motor_type_gas","motor_type_hybrid","motor_type_petrol","color_beige","color_black","color_blue","color_brown","color_cherry","color_clove","color_golden","color_gray","color_green","color_orange","color_other","color_pink","color_purple","color_red","color_silver","color_skyblue","color_white","type_Coupe","type_Universal","type_hatchback","type_minivan / minibus","type_pickup","type_sedan","type_suv"], "data": [[0.9166666666666643,0.019165922871590562,0.75,0.47368421052631576,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]]}}'