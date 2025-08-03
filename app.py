import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import string


tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("SMS SPAM OR HAM PREDICTOR")

def text_preprocessing(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
            if i  not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
                y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(nltk.stem.porter.PorterStemmer().stem(i))

    return " ".join(y)

imp=st.text_input("ENTER YOUR MSG HERE")
if st.button("PREDICT"):
    text=text_preprocessing(imp)

    inp_main=tfidf.transform([text])

    result=model.predict(inp_main)[0]
    if result == 1:
       st.text("SPAM PREDICTED")
    else:
       st.text("SPAM NOT PREDICTED")