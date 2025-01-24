import pandas as pd
df=pd.read_csv("house-prices.csv")
print(df)
dummi=pd.get_dummies(df.Neighborhood,dtype=int)
print(dummi)
merged=pd.concat([df,dummi],axis="columns")
print(merged)
final_data=merged.drop(["Neighborhood"],axis="columns")
print(final_data)
cols_to_use=["Price","SqFt","Bedrooms","Neighborhood"]
dt=df[cols_to_use]
print(dt.head())
print(dt.tail())
x=dt.drop("Price",axis=1)
y=dt["Price"]
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(final_data,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
abc=LinearRegression().fit(train_x,train_y)
abc.score(test_x,test_y)
print(abc)
from sklearn import linear_model
lone=linear_model.lasso(alpha=50,max_iter=100,tol=0.1)
