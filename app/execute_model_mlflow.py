
import pandas as pd
import joblib
from sklearn.linear_model._huber import HuberRegressor
from sklearn.ensemble._forest import ExtraTreesRegressor
from sklearn.linear_model._bayes import BayesianRidge
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import mlflow
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('app/data/data.csv')

def transform_running(run):
    if run[-2:]=='km':
        return float(run.replace('km',''))
    else:
        return float(run.replace('miles',''))*1.609344

df['running']=df['running'].apply(transform_running)
qual_mappings = {'excellent': 3, 'good':2, 'crashed': 0, 'normal': 1, 'new': 4}
df['status'] = df['status'].map(qual_mappings)
df=df.drop('wheel',axis=1)

encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['model', 'motor_type','color','type']])


encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['model', 'motor_type','color','type']))

df = pd.concat([df[['year','running','status','motor_volume','price']].reset_index(drop=True), encoded_df], axis=1)

df['motor_type_gas']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_gas'],axis=1)
df['motor_type_petrol']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_petrol'],axis=1)
df=df.drop(['motor_type_petrol and gas'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(['price'], axis=1), # predictive variables
    df['price'], # target
    test_size=0.1, # portion of dataset to allocate to test set
    random_state=0, # we are setting the seed here
)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns
)

joblib.dump(scaler, 'scaler_entrenado.pkl')
params = {
    "n_estimators": 100,
    "random_state": 123,
    "epsilon": 5,
    "max_iter": 100,
    "alpha": 0.01
}

# params = {
#     "n_estimators": 50,
#     "random_state": 30,
#     "epsilon": 10,
#     "max_iter": 200,
#     "alpha": 0.00001
# }


# 2. Instanciar los modelos en un diccionario
modelos = {
    "RandomForest": RandomForestRegressor(n_estimators=params["n_estimators"], random_state=params["random_state"]),
    "Huber": HuberRegressor(epsilon=params["epsilon"], max_iter=params["max_iter"], alpha=params["alpha"]),
    "BayesianRidge": BayesianRidge(),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=params["n_estimators"], random_state=params["random_state"])
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("car_values")
for nombre_modelo, modelo in modelos.items():
    
    with mlflow.start_run(run_name=nombre_modelo):

        
        mlflow.log_params(params)
        
        modelo.fit(X_train, y=Y_train)
        
        predicciones = modelo.predict(X_test)
        
        mse = mean_squared_error(Y_test, predicciones)
        r2 = r2_score(Y_test, predicciones)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(modelo, f"modelo_{nombre_modelo}")
        