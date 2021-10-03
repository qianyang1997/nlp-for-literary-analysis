import time
import spacy
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')


text = """
It is a truth universally acknowledged, that a single man
in possession of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may
be on his first entering a neighbourhood, this truth is so
well fixed in the minds of the surrounding families, that
he is considered as the rightful property of some one or other
of their daughters.

“My dear Mr. Bennet,” said his lady to him one day, “have you
heard that Netherfield Park is let at last?”

Mr. Bennet replied that he had not.

“But it is,” returned she; “for Mrs. Long has just been here,
and she told me all about it.”

Mr. Bennet made no answer.

“Do not you want to know who has taken it?” cried his wife 
impatiently.

“You want to tell me, and I have no objection to hearing it.”

This was invitation enough.

“Why, my dear, you must know, Mrs. Long says that Netherfield
is taken by a young man of large fortune from the north of
England; that he came down on Monday in a chaise and four to
see the place, and was so much delighted with it that he agreed
with Mr. Morris immediately; that he is to take possession
before Michaelmas, and some of his servants are to be in the
house by the end of next week.”
"""

def nltk_pipeline(text):

    # word tokenization
    words = word_tokenize(text)

    # pos tagging
    pos = pos_tag(words)

    # lemmatize
    words_lower = [word.lower() for word in words]
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(word) for word in words_lower]

    # remove stopwords
    stop_words = stopwords.words('english')
    words_no_stopwords = [word for word in words_lemmatized if word not in stop_words]

    return words, pos, words_lemmatized, words_no_stopwords


def spacy_pipeline(text):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # tokenization
    words = [token.text for token in doc]

    # pos tagging
    pos = [token.tag_ for token in doc]

    # lemmatize
    words_lemmatized = [token.lemma_ for token in doc]

    # remove stopwords
    stop_words = nlp.Defaults.stop_words
    words_no_stopwords = [word for word in words_lemmatized if word not in stop_words]

    return words, pos, words_lemmatized, words_no_stopwords

    
if __name__  == '__main__':

    # time it
    t0 = time.time()
    nw, np, nl, ns = nltk_pipeline(text)
    t1 = time.time()

    t2 = time.time()
    sw, sp, sl, ss = spacy_pipeline(text)
    t3 = time.time()

    print(f'nltk tokenization:\n'
          f'{nw}\n'
          f'spacy tokenization:\n'
          f'{sw}\n'
          f'nltk pos tagging:\n'
          f'{np}\n'
          f'spacy pos tagging:\n'
          f'{sp}\n'
          f'nltk lemmatization:\n'
          f'{nl}\n'
          f'spacy lemmatization:\n'
          f'{sl}\n'
          f'nltk stopword removal:\n'
          f'{ns}\n'
          f'spacy stopword removal:\n'
          f'{ss}\n'
          f'nltk time: {t1 - t0}; spacy time: {t3 - t2}')

    

    