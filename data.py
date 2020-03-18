import urllib.parse as parse
import pyodbc
import sqlalchemy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import string
import re
from bert import bert_tokenization
from sklearn.pipeline import Pipeline
import tqdm
import pickle as pkl
import os
import tfpipeline.v2 as tfp
from tensorflow import io


def eri_db(db, server='SNADSSQ3', driver='{SQL Server}'):
    params = ('Driver={driver};'
              'Server={server};'
              'Database={db};'
              'Trusted_Connection=yes;')
    params = params.format(driver=driver, server=server, db=db)
    params = parse.quote(params)
    uri = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
    return sqlalchemy.create_engine(uri)

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class DescriptionCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X):
        return self
    def transform(self, X):
        def clean(x):
            sep = ['*', '-', '>', '<', '~']
            other_punc = ''.join([s for s in string.punctuation if s not in sep])
            ## Normalize sentence separators
            pat = '[\\.\\?!;]{1,5}\s'
            x = re.sub(pat, '. ', x)

            ## Ensures that there is punctuation at the end of a line
            pat = '([^\\.])\s*(?:\r\n|\n)'
            x = re.sub(pat, '\\1. ', x)

            ## Strips out any bullet-like formatting
            pat = '\\.\s*[^\w\s{}]\s*(\w)'.format(other_punc)
            x = re.sub(pat, '. \\1', x)
            pat = ':\s*[^\w\s{}]\s*(\w)'.format(other_punc)
            x = re.sub(pat, '. \\1', x)
            pat = '(\w)\s*[^\w\s{}]\s*(\w)'.format(other_punc)
            x = re.sub(pat, '\\1. \\2', x)
            pat = '^\s*[^\w\s{}]\s*(\w)'.format(other_punc)
            x = re.sub(pat, '\\1', x)
            return x
        clean_desc = []
        for x in tqdm.tqdm(X, desc='DescriptionCleaner'):
            clean_desc.append(clean(x))
        return np.array(clean_desc)

class BERTTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_file, max_seq_len):
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.tokenizer = bert_tokenization.FullTokenizer(vocab_file)
    def fit(self, X):
        return self
    def transform(self, X):
        def tokenize(x):
            x = self.tokenizer.tokenize(x)
            x = ['[CLS]'] + x + ['[SEP]']
            x = self.tokenizer.convert_tokens_to_ids(x)
            if len(x) > self.max_seq_len:
                x = x[:self.max_seq_len]
            while len(x) < self.max_seq_len:
                x.append(0)
            return x
        token_ids = []
        for x in tqdm.tqdm(X, desc='BERTTokenizer'):
            token_ids.append(tokenize(x))
        return np.array(token_ids)

if __name__ == '__main__':
    assessor_work = eri_db('AssessorWork')

    with open('get_desc_data.sql', 'r') as f:
        sql = f.read()
    df = pd.read_sql(sql, assessor_work, index_col='job_id').sort_index()
    job_ids = df.index.values
    y = df.flsa.values

    desc_pipe = Pipeline([
        ('df_extract', DataFrameTransformer('title_desc')),
        ('cleaner', DescriptionCleaner()),
        ('tokenizer', BERTTokenizer(os.path.join(os.getcwd(), 'data', 'uncased_base', 'vocab.txt'),
                                    max_seq_len=256))
    ])
    X = desc_pipe.fit_transform(df)
    X = np.array(X).reshape(-1, 256)
    y = np.array(y, dtype='int64')
    shuf_idx = np.random.permutation(len(y))
    X = X[shuf_idx]
    y = y[shuf_idx]

    assert len(bert_input) == len(y)
    splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=43)
    for train_index, test_index in splitter.split(X, y):
        X_tmp, X_test = X[train_index], X[test_index]
        y_tmp, y_test = y[train_index], y[test_index]
    splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=12)
    for train_index, valid_index in splitter.split(X_tmp, y_tmp):
        X_train, X_valid = X_tmp[train_index], X_tmp[valid_index]
        y_train, y_valid = y_tmp[train_index], y_tmp[valid_index]
    ds_lookup = {
        'X_train': X_train,
        'X_test': X_test,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_test': y_test,
        'y_valid': y_valid
    }
    for file, obj in ds_lookup.items():
        path = os.path.join('data', 'bert_only', '{}.npy'.format(file))
        with open(path, 'wb') as f:
            np.save(f, obj)

    io.gfile.makedirs('saved_models')
    file = os.path.join('saved_models', 'scaler.pkl')
    with open(file, 'wb') as f:
        pkl.dump(scaler, f)
