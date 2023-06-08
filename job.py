import joblib
from sklearn.tree import DecisionTreeClassifier

model=joblib.load( 'our_pridction.joblib')
predictions= model.predict([[21,1]])
print(predictions)
