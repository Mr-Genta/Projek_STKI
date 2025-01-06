# Importing essential libraries
from flask import Flask, render_template, request
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt_tab')

# Label Encoder use to Encode target labels with value between 0 and n_classes-1
Encoder = LabelEncoder()

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


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
        result = int(output_array[0])
        
        
        return render_template('result.html', prediction=result, naskah=berita)
        
        

if __name__ == '__main__':
	app.run(debug=True)

