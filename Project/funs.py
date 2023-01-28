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


class TextTransformer(Word2Vec):
    def __init__(self, data, vector_size=20, min_count=1) -> None:
        self.data = data
        super().__init__(data, vector_size=vector_size, min_count=min_count)

    def get_agg_word2vec(self, text, agg_func=np.mean):
        # Get the word2vec representation of each word in the text
        word_vectors = np.array([self.wv[word] for word in text if word in self.wv.index_to_key])
        res = agg_func(word_vectors, axis=0)
        if np.isnan(res).any():
            res = np.zeros(self.vector_size)
        return res

    def transform_data(self, column_name, data=None):
        if data is None:
            data = self.data
        n = data.shape[0]
        print(f'Transforming {column_name} data should take around {(n / 90 / 60):3f} minutes')
        X_train_vectors = data.apply(lambda x: self.get_agg_word2vec(x))
        X_train_vectors = np.array(X_train_vectors)
        X_train_vectors = np.vstack(X_train_vectors)
        df_train = pd.DataFrame(X_train_vectors,
         columns=['num_' + column_name + '_' + str(nr) for nr in np.arange(20)])
        self.data_transformed = df_train
        return df_train
    