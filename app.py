import streamlit as st
import pickle
import numpy as np

# Load the model and vectoriser
model = pickle.load(open('Sentiment Analysis/model.pkl', 'rb'))
vectorizer = pickle.load(open('Sentiment Analysis/vectorizer.pkl', 'rb'))


# Streamlit UI
st.markdown("<h1 style='text-align: center; color: blue;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
st.image('Sentiment Analysis/bg.jpg', use_container_width=True)

# Input text from the user
user_input = st.text_area("Enter text for sentiment analysis:", height=200)

if st.button("Analyze", key="analyze_button"):
    if user_input:
        # Preprocess the input text
        input_data = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_data)
        
        # Output the result
        if prediction[0] == 1:
            st.success("The sentiment is Positive!", icon="✅")
        elif prediction[0] == 0:
            st.success("The sentiment is Neutral", icon="⚪")
        else:
            st.error("The sentiment is Negative.", icon="❌")
    else:
        st.warning("Please enter some text before analyzing.")
