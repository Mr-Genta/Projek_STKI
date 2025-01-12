# Importing essential libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import string
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# nltk.download('stopwords')
nltk.download('punkt')

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# def clean(text):
#     # lowercase text
#     text = text.lower()
    
#     # clear URL, mention, hashtag, alpahanumeric
#     text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split())
    
#     # remove punctuation
#     punct = set(string.punctuation) 
#     text = "".join([ch for ch in text if ch not in punct])
    
#     return text

# def tokenize(text):
#     return word_tokenize(text)

# def remove_stop_words(text):
#     stop_words =  set(stopwords.words('indonesian'))
#     word_tokens_no_stopwords = [w for w in text if not w in stop_words]
#     return word_tokens_no_stopwords

# def preprocess(text):
#     text = clean(text)
#     text = tokenize(text)
#     text = remove_stop_words(text)
#     return text

# Load Model
filename = 'model_berita.pkl'
model = pickle.load(open(filename, 'rb'))

# Load vectorizer
filename2 = 'tfidf_vectorizer.pkl'
vectorizer = pickle.load(open(filename2, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        berita = request.form.get('deteksi')
        lower = stemmer.stem(berita.lower())
        tokens = word_tokenize(lower)
        x = vectorizer.transform(tokens)
        output_array = model.predict(x)
        print(output_array)
        result = output_array
        
        
        return render_template('result.html', prediction=result, naskah=berita)
        
        

if __name__ == '__main__':
	app.run(debug=True)

