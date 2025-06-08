import streamlit as st
import pandas as pd
from data_processing import TextPreprocessor, Word2VecVectorizer
import pickle

st.set_page_config(page_title="Phishing Email Classifier", layout="wide")
st.title("Phishing Email Classifier")
st.write("Insira o texto do email abaixo para verificar se é phishing ou não.")

with st.form(key='email_form'):
    email_text = st.text_area("Texto do Email", height=300)
    submit_button = st.form_submit_button(label='Classificar')

if submit_button:
    if email_text:
        # processamento do texto
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(email_text)
        vectorizer = Word2VecVectorizer(model_path="models/models_w2v/vmodelo_w2v.model")
        X = vectorizer.vectorize_texts([processed_text])
        
        # carregamento do modelo e previsão
        with open("models/models_phishing/model_phishing.pkl", "rb") as f:
            model = pickle.load(f)
        
        prob = model.predict(X)[0]
        pred = (prob > 0.5).astype(int).flatten()  

        if pred == 1:
            st.error(f"Este email é phishing! (Chance de ser phishing: {prob[0]:.2f})")
        else:
            st.success(f"Este email não é phishing! (Chance de ser phishing: {prob[0]:.2f})")
    else:
        st.warning("Por favor, insira o texto do email para classificação.")

st.sidebar.header("Instruções")
st.sidebar.write("""
Para usar este classificador de phishing, insira o texto do email (em inglês) que deseja verificar. 
O modelo irá analisar o conteúdo e informar se o email é potencialmente phishing ou não.""")

st.sidebar.header("Sobre o Modelo")
st.sidebar.write("""
Este classificador foi treinado usando um modelo Word2Vec para vetorização de texto e um modelo 
de classificação para identificar emails de phishing. O modelo foi treinado com um conjunto 
de dados contendo exemplos de emails phishing e não phishing.""")
