import pandas as pd
import joblib 
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

df=pd.read_csv('data/data.csv')

x=df.drop('median_house_value',axis=1)
x=pd.get_dummies(x)
x=x.fillna(x.median())
y=df['median_house_value']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=joblib.load('linear_regression_model.pkl')

predictions=model.predict(x_test)

rmse=mse(y_test,predictions)
print("RMSE:",rmse)