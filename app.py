#Importing the Libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import pickle
import flask
import newspaper
from newspaper import Article
import urllib
import numpy as np

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model_tfidf_azure.pkl', 'rb') as handle:
	tfidf_model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('home.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/more',methods=['GET','POST'])
def more():
    return render_template('more.html')
@app.route('/predict',methods=['GET','POST'])

def predict():
    url =request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    print("tfidf",url)
    
    article = Article(str(url))
 
    article.download()
    article.parse()
    article.nlp()
    title = article.title

    print(title)
    news = article.summary
    #X_input = TfidfVectorizer.fit_transform([news])
    #Fin=TfidfVectorizer.get_feature_names_out(X_input)
    #print("FINNNN tfidf", Fin)
    #Passing the news article to the model and returing whether it is Fake or Real
    prediction = tfidf_model.predict([news])
    
    # Predict the class of the input news article
    #prediction = tfidf_model.predict([news])

# Get the class probabilities for the input news article
    probabilities = tfidf_model.predict_proba([news])
    print(probabilities)

# Get the confidence level of the predicted class
    confidence_level = probabilities[0][prediction[0]]

    #print(k)
    #Passing the news article to the model and returing whether it is Fake or Real
    #pred = model.predict([news])
    
    # Get the class probabilities for the input news article
    #probabilities = model.predict_proba([news])
    #print("The predicted class is:", prediction[0])
    class_labels = {1: 'Fake', 0: 'Real'}
    predicted_class = class_labels[prediction[0]]
    print("The predicted class is:", predicted_class)
    print("The confidence level of the predicted class is:", confidence_level)

    return render_template('predict.html', prediction_text2=' Confidence level behind this prediction is: {:.2f}'.format(confidence_level), prediction_text='Our model predicts the news you searched titled', title= '"{}"'.format(title), prediction_text1=predicted_class, prediction_text3=' A glimpse of the news: {}'.format(news))
    #return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))
    

with open('count_small_model.pkl', 'rb') as handle:
	count_model = pickle.load(handle)
@app.route('/tfidf',methods=['GET','POST'])
def tfidf():
    url =request.get_data(as_text=True)[5:]
    print(url)
    
    url = urllib.parse.unquote(url)
    print("count",url)
    article = Article(str(url))
    article.download()
    #print(art1)
    article.parse()
    article.nlp()
    title = article.title

    print(title)
    #print(art2)
    news = article.summary
    #print(news)c
    #print("Article's Keywords:\n")c
    k=article.keywords
    #X_input = CountVectorizer.fit_transform([news])
    #Fin=CountVectorizer.get_feature_names_out([news])
    #print("FINNNN tfidf", Fin)
    # Predict the class of the input news article
    prediction = count_model.predict([news])

# Get the class probabilities for the input news article
    probabilities = count_model.predict_proba([news])
    print(probabilities)

# Get the confidence level of the predicted class
    confidence_level = probabilities[0][prediction[0]]

    #print(k)
    #Passing the news article to the model and returing whether it is Fake or Real
    #pred = model.predict([news])
    
    # Get the class probabilities for the input news article
    #probabilities = model.predict_proba([news])
    #print("The predicted class is:", prediction[0])
    class_labels = {1: 'Fake', 0: 'Real'}
    predicted_class = class_labels[prediction[0]]
    print("The predicted class is:", predicted_class)
    print("The confidence level of the predicted class is:", confidence_level)
    return render_template('predict.html', prediction_text2=' Confidence level behind this prediction is: {:.2f}'.format(confidence_level), prediction_text='Our model predicts the news you searched titled', title= '"{}"'.format(title), prediction_text1=predicted_class, prediction_text3=' A glimpse of the news: {}'.format(news))

    #return render_template('predict.html', prediction_text2=' Confidence level behind this prediction is: {: .2f}'.format(confidence_level), prediction_text='Our model predicts the news you searched titled "{}" is'.format(title), prediction_text1= ' "{}"'.format(predicted_class),prediction_text3=' A glipmse of the news: {}'.format(news))
    #return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0])))
    #The probability of occuring these combinations of word is high in Real news.
    # #return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))
    #return render_template('main.html', prediction_text2=' Confidence level behind this prediction is: "{: .2f}"'.format(confidence_level), prediction_text='Our model predicts the news is "{}"'.format(predicted_class))
    #return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))

# Define the news article to predict
news_article = "Ever get the feeling your life circles the roundabout rather than heads in a straight line toward the intended destination? [Hillary Clinton remains the big woman on campus in leafy, liberal Wellesley, Massachusetts"


if __name__=="__main__":
    #port=int(os.environ.get('PORT',5000))
    app.run(debug=True,use_reloader=False)
    #print(news[0])
