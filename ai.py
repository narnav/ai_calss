from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# loading data
music_dt  =pd.read_csv('music.csv')

# print(music_dt.tail())

X=music_dt.drop(columns=['genre']) # sample features
Y=music_dt['genre'] # sample output

model = DecisionTreeClassifier()
model.fit(X,Y) # load features and sample data
joblib.dump(model, 'our_pridction.joblib') #binary file

predictions= model.predict([[21,1],[22,0]]) # make prediction base on the 
print(predictions)