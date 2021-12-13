import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

house = pd.read_csv("house_price.csv")

house.rename(columns = {'Sq.ft':'Sq_ft'},inplace = True)
house.rename(columns = {'Old(years)':'Old'},inplace = True)

dummies = pd.get_dummies(house.Location)
house = pd.concat([dummies,house],axis = 'columns')

house.drop(['Location'],axis = 'columns' ,inplace =True)


x = house[['Bommanahalli', 'Whitefield', 'BHK', 'Furnishing', 'Sq_ft', 'Old',
       'Floor']]
y = house['Price']


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 42)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
predictions = lm.predict(x_test)


from sklearn.metrics import r2_score
r2_score(y_test,predictions)


pickle.dump(lm,open('model.pkl','wb'))


model=pickle.load(open('model.pkl','rb'))