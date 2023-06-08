from flask import Flask,render_template,request
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from csv import writer

api = Flask(__name__)


@api.route('/')
def hello():
    return render_template('index.html')

@api.route('/pred', methods=['post'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    # prediction
    music_dt  =pd.read_csv('music.csv')
    X=music_dt.drop(columns=['genre']) # sample features
    Y=music_dt['genre'] # sample output
    # print(music_dt)
    model = DecisionTreeClassifier()
    model.fit(X,Y) # load features and sample data
    print( [age,gender])
    predictions= model.predict([[age,gender]]) # make prediction base on the 
    print(predictions)
    return render_template('pred.html',msg=predictions)

@api.route('/learn', methods=['post','get'])
def learn():
    if request.method=="get":
        return render_template('learn.html')
    # post
    age = request.form.get('age')
    gender = request.form.get('gender')
    genre = request.form.get('genre')
    lst =[age,gender,genre]
    with open('music.csv', 'a', newline='') as file:
        mwriter = writer(file)
        mwriter.writerow(lst)
    # with open('music.csv', 'a') as f_object:
    #     writer_object = writer(f_object)
    #     writer_object.writerow(List)
    #     f_object.close()
    return render_template('learn.html')

if __name__ == '__main__':
    api.run(debug=True)