import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('app/data/batch.csv')

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

df = pd.concat([df[['year','running','status','motor_volume']].reset_index(drop=True), encoded_df], axis=1)

df['motor_type_gas']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_gas'],axis=1)
df['motor_type_petrol']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_petrol'],axis=1)
df=df.drop(['motor_type_petrol and gas'], axis=1)


scaler_guardado = joblib.load('scaler_entrenado.pkl')

columnas_esperadas = scaler_guardado.feature_names_in_

df = df.reindex(columns=columnas_esperadas, fill_value=0)

X_batch_escalado = pd.DataFrame(
    scaler_guardado.transform(df),
    columns=df.columns
)

X_batch_escalado.to_csv('app/data/batch_pred_procesado.csv', index=False)
print("¡Archivo batch generado y alineado correctamente!")