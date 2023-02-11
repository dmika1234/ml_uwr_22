import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


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
        text_data = text_data[column_names].copy()
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


class BagofWords:
    def __init__(self, text_data, text_colnames) -> None:
        self.text_data = text_data
        self.text_colnames = text_colnames
        self.all_words = {}
        self.most_popular_words = {}
        self.nr_of_words = 1000
        self.onehot_dfs = {}
    def get_all_words(self):
        for column_name in self.text_colnames:
            self.all_words[column_name] = np.concatenate(self.text_data[column_name])

    def get_most_pop_words(self, nr_of_words = 1000):
        self.nr_of_words = nr_of_words
        for column_name in self.text_colnames:
            count_words = Counter(self.all_words[column_name])
            self.most_popular_words[column_name] = np.array(count_words.most_common(self.nr_of_words))[:,0]
    
    def prepare_onehot_dfs(self):
        for text_colname in self.text_colnames:
            onehot_array = np.zeros((self.text_data.shape[0], self.nr_of_words))
            for i in np.arange(self.text_data.shape[0]):
                onehot_array[i] = np.isin(self.most_popular_words[text_colname], self.text_data[text_colname][i]).astype('int64')
            column_names = np.vectorize(lambda x: text_colname + '_' + str(x))(self.most_popular_words[text_colname])
            self.onehot_dfs[text_colname] = pd.DataFrame(onehot_array, columns=column_names)

    def get_onehot_dfs(self):
        X = pd.DataFrame()
        for text_colname in self.text_colnames:
            X = pd.concat((X, self.onehot_dfs[text_colname]), axis=1)
        return X
    
    def encode_onehot(self, nr_of_words = 1000):
        self.get_all_words()
        self.get_most_pop_words(nr_of_words = 1000)
        self.prepare_onehot_dfs()
        return self.get_onehot_dfs()



class TfidTranformer:
    def __init__(self, number_of_vars = 50):
        self.number_of_vars = number_of_vars
        self.X_tfdif = pd.DataFrame()
    
    def vectorize_transform(self, text_data_str, train_indcs, text_colnames):
        for colname in text_colnames:
            vectorizer = TfidfVectorizer(max_features=number_of_vars, min_df=2).fit(text_data_str[colname][train_indcs])
            df_transformed = pd.DataFrame(vectorizer.transform(text_data_str[colname]).toarray(), 
            columns=['num_' + colname + '_' + str(nr) for nr in np.arange(number_of_vars)])
            print(f'{colname} data successfuly transformed!')
            self.X_tfdif = pd.concat((self.X_tfdif, df_transformed), axis=1)
        return self.X_tfdif


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
         columns=['num_' + column_name + '_' + str(nr) for nr in np.arange(self.vector_size)])
        self.data_transformed = df_train
        return df_train


class GoogleTextTransformer:
    def __init__(self, path_to_model='GoogleNews-vectors-negative300.bin') -> None:
        self.wv = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
        self.vector_size = self.wv.vector_size

    def get_agg_word2vec(self, text, agg_func=np.mean):
        # Get the word2vec representation of each word in the text
        word_vectors = np.array([self.wv[word] for word in text if word in self.wv.index_to_key])
        res = agg_func(word_vectors, axis=0)
        if np.isnan(res).any():
            res = np.zeros(self.vector_size)
        return res

    def transform_data(self, column_name, data=None, verbose=False):
        n = data.shape[0]
        if verbose:
            print(f'Transforming {column_name} data should take around {(n / 90 / 60):3f} minutes')
        X_train_vectors = data.apply(lambda x: self.get_agg_word2vec(x))
        X_train_vectors = np.array(X_train_vectors)
        X_train_vectors = np.vstack(X_train_vectors)
        df_train = pd.DataFrame(X_train_vectors,
         columns=['num_' + column_name + '_' + str(nr) for nr in np.arange(self.vector_size)])
        self.data_transformed = df_train
        return df_train   
    

def get_train_test_indcs(X, y, test_size, random_state, stratify):
    X_train, X_test = train_test_split(X, y, test_size=test_size,
     random_state=random_state, stratify=stratify)[0:2]
    return X_train.index, X_test.index


def evaluate_performance(y_test, y_pred, prob=False, threshold=0.2, round=4, save_pred=False):
    if prob:
        y_pred = (y_pred >= threshold).astype('int64')
    results = {}
    if save_pred:
        results['y_pred'] = y_pred
    # Get the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate the detection percentage
    results['detection_percentage'] = np.round(tp / (tp + fn), round)
    # Calculate precision
    results['precision'] = np.round(tp / (tp + fp), round)
    # Calculate accuracy
    results['accuracy'] = np.round((tp + tn) / (tp + tn + fp + fn), round)
    # Calculate F1-score
    results['f1_score'] = np.round(2 * (results['precision'] * results['detection_percentage']) /\
                          (results['precision'] + results['detection_percentage']), round)
    results['auc_roc'] = np.round(roc_auc_score(y_test, y_pred), round)

    return results
