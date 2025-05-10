# step1 : import all the libraries and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}
# Load the Pre-trained model with ReLU activation
model=load_model('Simple_rnn_imdb.h5')

# step2: Helper functions
#function to decode reviews

def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encode_review])

# Function to preprocess user input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#step3: Prediction function

def predict_sentiment(review):
    pre_processed_input=preprocess_text(review)

    prediction=model.predict(pre_processed_input)

    sentiment = 'positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment, prediction[0][0]

import streamlit as st
## Desgning the streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#user input

user_input= st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input =  preprocess_text(user_input)

    ## Make Prediction

    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    # Display the results

    st.write(f'sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a Movie Review.')