import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
import pickle

class TextPreprocessor:
    def __init__(self):
        pass

    def _remove_accents(self, text):
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd_form if not unicodedata.category(c) == "Mn"])

    def _remove_stopwords(self, text):
        return " ".join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = self._remove_accents(text)
        text = self._remove_stopwords(text)
        return text.strip()

class Word2VecVectorizer:
    def __init__(self, model_path="modelos_w2v/vmodelo_w2v.model"):
        self.model = Word2Vec.load(model_path)
        self.vector_size = self.model.vector_size

    def vetor_medio(self, texto):
        palavras = texto.split()
        vetores = [self.model.wv[p] for p in palavras if p in self.model.wv]
        return np.mean(vetores, axis=0) if vetores else np.zeros(self.vector_size)

    def vectorize_texts(self, textos):
        vetores = [self.vetor_medio(t) for t in textos]
        return pd.DataFrame(vetores, columns=[f"w2v_{i}" for i in range(self.vector_size)])




