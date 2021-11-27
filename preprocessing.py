from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

def read_data(filepath):
    with open(filepath, 'r') as f:
        text = f.readlines()
    text = [t for t in text if len(t) >= 100]
    return text


def create_dataset(barack_filepath, michelle_filepath):
    barack = read_data(barack_filepath)
    michelle = read_data(michelle_filepath)
    b_labels = [0] * len(barack)
    m_labels = [1] * len(michelle)
    return barack, b_labels, michelle, m_labels


def create_train_test(b_filepath, m_filepath,
                      b_test_size, m_test_size,
                      b_seed, m_seed):
    
    b, bl, m, ml = create_dataset(b_filepath, m_filepath)
    bx_train, bx_test, by_train, by_test = train_test_split(b, bl,
                                                            test_size=b_test_size,
                                                            random_state=b_seed)
    mx_train, mx_test, my_train, my_test = train_test_split(m, ml,
                                                            test_size=m_test_size, 
                                                            random_state=m_seed)
    x_train = bx_train + mx_train
    x_test = bx_test + mx_test
    y_train = by_train + my_train
    y_test = by_test + my_test
    return x_train, x_test, y_train, y_test


def eda(text_list):
    print(f"Number of paragraphs: {len(text_list)}")
    tokenizer = RegexpTokenizer('[A-Za-z-]+')

    while True:
        try:
            stops = stopwords.words('english')
            break
        except LookupError:
            nltk.download('stopwords')
    text = ' '.join(text_list)
    tokens = tokenizer.tokenize(text)
    tokens = [t.lower() for t in tokens]
    nonstops = [t for t in tokens if t not in stops]
    print(f"Number of words: {len(tokens)}")
    print(f"Number of content words: {len(nonstops)}")

    dic = dict(Counter(nonstops))
    df = pd.DataFrame({'word': dic.keys(), 'count': dic.values()})
    return df.sort_values('count', ascending=False)
    



if __name__ == '__main__':

    b, _, m, _ = create_dataset('data/barack.txt', 'data/michelle.txt')
    _, x_test, _, y_test = create_train_test('data/barack.txt',
                                        'data/michelle.txt',
                                        0.2, 0.2, 42, 21)
    df = pd.DataFrame({'text': x_test, 'label': y_test})
    print(sum(df.label==0))
    print(sum(df.label==1))
    #eda(b).to_csv('top_barack_words.csv', index=False)
    #eda(m).to_csv('top_michelle_words.csv', index=False)