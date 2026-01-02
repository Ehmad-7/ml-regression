import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df=pd.read_csv('data/data.csv')
print(df.head())
print(df.info())



x=df.drop('median_house_value',axis=1)
x=pd.get_dummies(x)
x=x.fillna(x.median())
y=df['median_house_value']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

joblib.dump(model,'linear_regression_model.pkl')
print("Model trained and saved successfully!")