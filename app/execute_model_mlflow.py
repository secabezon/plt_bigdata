
import pandas as pd

from sklearn.linear_model._huber import HuberRegressor
from sklearn.ensemble._forest import ExtraTreesRegressor
from sklearn.linear_model._bayes import BayesianRidge
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

df=pd.read_csv('data/train.csv')

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

Rdm_frst=RandomForestRegressor(n_estimators=100,random_state=123)
Rdm_frst.fit(X_train,y=Y_train)

pred_rf = Rdm_frst.predict(X_test)

HbrRgrssr=HuberRegressor(epsilon=5, max_iter=100, alpha=0.01)
HbrRgrssr.fit(X_train,y=Y_train)

pred_hbr = HbrRgrssr.predict(X_test)

ByssnRdg=BayesianRidge()
ByssnRdg.fit(X_train,y=Y_train)

pred_bysn = ByssnRdg.predict(X_test)

EtrsRgsr=ExtraTreesRegressor(n_estimators=100, random_state=123)
EtrsRgsr.fit(X_train,y=Y_train)

pred_extr = EtrsRgsr.predict(X_test)


print('Finaliza bien')