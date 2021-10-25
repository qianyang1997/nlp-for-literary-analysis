import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# clean json
def get_data():
    with open('yelp_subset.json', 'r') as raw:
        line = raw.readlines()
    data = [json.loads(l) for l in line]
    items = [(data[i]['text'], data[i]['stars']) for i in range(len(data))]
    text, star = list(zip(*items))
    return pd.DataFrame({'text': text, 'stars': star})

# vectorize top 100 words or 2-grams (other than stop words) into features 
def vectorize(array, ngram, max_df, max_features):
    nrange = (ngram,) * 2
    vectorize = CountVectorizer(ngram_range=nrange, max_df=max_df,
                                max_features=max_features)
    features = vectorize.fit_transform(array)
    return features

# calculate # documents
def num_doc(data):
    return data.shape[0]

# calculate # labels
def num_labels(data, label='stars'):
    return data[label].nunique()

# calculate label distribution
def dist_labels(data, label='stars'):
    avg = data[label].mean()
    maxStar = data[label].max()
    minStar = data[label].min()
    med = data[label].median()
    sd = data[label].std()
    q25 = data[label].quantile(0.25)
    q75 = data[label].quantile(0.75)
    return avg, maxStar, minStar, med, sd, q25, q75

# calculate avg word length
def avg_word(data, col='text'):
    length = []
    for doc in data[col]:
        tokens = word_tokenize(doc)
        length += [len(tokens)]
    return np.mean(length)

# put everything in a table
def summary(data, col='text', label='stars'):
    label_distributions = dist_labels(data, label)
    return pd.DataFrame.from_dict(
        {
            'number_of_documents': [num_doc(data)],
            'number_of_labels': [num_labels(data)],
            'label_mean': [label_distributions[0]],
            'label_median': [label_distributions[3]],
            'label_min': [label_distributions[2]],
            'label_max': [label_distributions[1]],
            'label_std': [label_distributions[4]],
            'label_25_quantile': [label_distributions[5]],
            'label_75_quantile': [label_distributions[6]]
        }, orient='index', columns=['value']
    )

# train test split - fix random seed
def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test

# preprocess data for models
def preprocess(ngram, max_df=0.8, max_features=100):

    data = get_data()
    features = vectorize(data['text'], ngram, max_df, max_features).toarray()
    x_train, x_test, y_train, y_test = split(features, data['stars'])
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    a, b, c, d = preprocess(1)
    print(a[:5], b[:5], c[:5], d[:5])