import re
import pandas as pd
import spacy
from gensim.models import Word2Vec


def clean_data(lpath, new_path):
    with open(lpath, 'r') as file:
        text = file.readlines()
    new_text = []
    for i, t in enumerate(text):
        if re.match('.*\t.*\t.*\t.*\t.*\t.*', t):
            print(i, t)
        if re.match('.*\t\s+\*.*', t):
            new_text += [re.sub('\t\s+\*', '', t)]
        else:
            new_text += [t]
    new_text = '\n'.join(new_text)
    with open(new_path, 'w+') as file:
        file.write(new_text)
        

def read_data(cpath, lpath):
    char = pd.read_csv(cpath, sep='\t', names=['cid', 'char', 'mid', 'movie', 'gender', 'credit'])
    lines = pd.read_csv(lpath, sep='\t', names=['lid', 'cid', 'mid', 'char', 'line'])
    return char.dropna().reset_index(drop=True), lines.dropna().reset_index(drop=True)


def get_lines(char, lines):
    # join character metadata to lines
    char2 = char.copy()
    char2.gender.replace({'M': 'm', 'F': 'f'}, inplace=True)
    char2 = char2[char2.gender != '?']
    joined = lines[['cid', 'line']].merge(char2[['cid', 'gender']], on='cid', how='inner').drop(columns='cid')
    # subset female and male lines
    f = joined[joined.gender == 'f']
    m = joined[joined.gender == 'm']
    print(f'Number of female lines: {f.shape[0]}')
    print(f'Number of male lines: {m.shape[0]}')
    return f.line, m.line


def tokenize(lines):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    listOfLists = []
    iter = 0

    for line in lines:
        iter += 1
        doc = nlp(line)
        words = [token.text.lower() for token in doc]
        if len(words) > 2:
            listOfLists += [words]
        if iter % 1000 == 0:
            print(f"Finished tokenizing {iter} lines.")

    print("Finished tokenizing.")
    return listOfLists


def list_to_file(listOfLists, filepath="tokens.txt"):
    """save tokenized list of list of str to txt file"""
    with open(filepath, "w") as f:
        for lst in listOfLists:
            for string in lst:
                f.write(string + " ")
            f.write("\n")


def load_tokens(filepath="tokens.txt"):
    """load tokens for training"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [article.split(" ") for article in lines]


def train_model(input_file, save_name="", load_name="", epochs=20, **kwargs):
    if load_name:
        model = Word2Vec.load(load_name)
    else:
        model = Word2Vec(**kwargs)
    model.build_vocab(input_file)
    model.train(input_file, total_examples=model.corpus_count, epochs=epochs)
    if save_name:
        model.save(save_name)
    return model


def test_model(input, model=None, load_name=""):
    if load_name:
        model = Word2Vec.load(load_name)
    if model:
        # input 1 is to input 2 as input 0 is to ...
        print(model.wv.most_similar(input, topn=5))


if __name__ == '__main__':
    # preprocessing and tokenize - don't run a second time
    """
    c, l = read_data('movie_characters_metadata.tsv', 'movie_lines2.tsv')
    f_line, m_line = get_lines(c, l)
    f_t, m_t = tokenize(f_line), tokenize(m_line)
    list_to_file(f_t, "f_line_tokens.txt")
    list_to_file(m_t, "m_line_tokens.txt")
    """

    # train model - feel free to comment out
    """
    f_t = load_tokens("f_line_tokens.txt")
    m_t = load_tokens("m_line_tokens.txt")
    params = {
        'min_count': 5,
        'vector_size': 100,
        'negative': 10,
        'workers': 10,
        'window': 5,
        'sg': 0
    }
    f_model = train_model(f_t, save_name="female_lines.model", epochs=20)
    m_model = train_model(m_t, save_name="male_lines.model", epochs=20)
    """

    input = ["house"]

    f_test = test_model(input=input, load_name="female_lines.model")
    m_test = test_model(input=input, load_name="male_lines.model")


