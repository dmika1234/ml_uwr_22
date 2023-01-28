import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def join_fun(list, sep=' '):
    text = sep.join(list)
    return text


class DataPreprocessor:
    def __init__(self) -> None:
        self.tokenizer = RegexpTokenizer(r'\w+')
        nltk.download('stopwords') 
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def preprocess_data(self, text_data, column_names, vectorize_fun=list):
        text_data = text_data[column_names]
        text_data.fillna('', inplace=True)
        
        for i in np.arange(text_data.shape[0]):
            for col in column_names:
                text_data.at[i,col] = self.tokenizer.tokenize(text_data.at[i,col].lower()) # removing punctuation
                filtered_sentence = []
                for w in text_data.at[i,col]:
                    if w not in self.stop_words: # removing words such as “the”, “a”, “an”, “in”
                        filtered_sentence.append(self.lemmatizer.lemmatize(w)) # grouping together the different inflected forms of a word 
                text_data.at[i,col] = vectorize_fun(filtered_sentence)

        return text_data

