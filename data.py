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

    with open('get_exp_data.sql', 'r') as f:
        sql = f.read()
    exp_df = pd.read_sql(sql, assessor_work, index_col='job_id')
    with open('get_rev_data.sql', 'r') as f:
        sql = f.read()
    rev_df = pd.read_sql(sql, assessor_work)
    rev_df = rev_df.pivot('job_id', 'cut_point', 'comp')
    rev_df.columns = ['low_comp', 'high_comp']
    comp_df = pd.concat([exp_df, rev_df])

    with open('get_desc_data.sql', 'r') as f:
        sql = f.read()
    desc_df = pd.read_sql(sql, assessor_work, index_col='job_id').sort_index()
    desc_df = desc_df[desc_df.index.isin(comp_df.index)]
    desc_job_ids = desc_df.index.values

    lb = LabelBinarizer()
    desc_y = lb.fit_transform(desc_df.flsa.values)
    classes = lb.classes_
    file = os.path.join(os.getcwd(), 'data', 'classes.npy')
    with open(file, 'wb') as f:
        np.save(f, classes)

    desc_pipe = Pipeline([
        ('df_extract', DataFrameTransformer('title_desc')),
        ('cleaner', DescriptionCleaner()),
        ('tokenizer', BERTTokenizer(os.path.join(os.getcwd(), 'data', 'uncased_base', 'vocab.txt'),
                                    max_seq_len=256))
    ])
    desc_X = desc_pipe.fit_transform(desc_df)

    bert_input = []
    comp_input = []
    y = []
    for i, id in enumerate(desc_job_ids):
        label = desc_y[i]
        input_ids = desc_X[i]
        low_comp = comp_df.loc[id, 'low_comp']
        high_comp = comp_df.loc[id, 'high_comp']
        rand_comps = np.random.randint(low_comp, high_comp, 10)
        comp_samples = [low_comp, high_comp, *rand_comps]
        for comp in comp_samples:
            bert_input.append(input_ids)
            comp_input.append(comp)
            y.append(label)
    bert_input = np.array(bert_input).reshape(-1, 256)
    comp_input = np.array(comp_input).reshape(-1, 1)
    y = np.array(y, dtype='int64')
    shuf_idx = np.random.permutation(len(y))
    bert_input = bert_input[shuf_idx]
    comp_input = comp_input[shuf_idx]
    y = y[shuf_idx]
    scaler = StandardScaler()
    comp_input = scaler.fit_transform(comp_input)

    assert len(bert_input) == len(comp_input) == len(y)
    splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=43)
    for train_index, test_index in splitter.split(comp_input, y):
        comp_input_train, comp_input_test = comp_input[train_index], all_comp[test_index]
        bert_input_train, bert_input_test = bert_input[train_index], all_input_ids[test_index]
        y_train, y_test = y[train_index], y[test_index]
    ds_lookup = {
        'comp_input_train': comp_input_train,
        'comp_input_test': comp_input_test,
        'bert_input_train': bert_input_train,
        'bert_input_test': bert_input_test,
        'y_train': y_train,
        'y_test': y_test
    }
    for file, obj in ds_lookup.items():
        path = os.path.join('data', '{}.npy'.format(file))
        with open(path, 'wb') as f:
            np.save(f, obj)

    io.gfile.makedirs('saved_models')
    file = os.path.join('saved_models', 'scaler.pkl')
    with open(file, 'wb') as f:
        pkl.dump(scaler, f)
