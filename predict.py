import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import dataset_stats as ds

# train model on entire yelp subset
def train_model():
    data = ds.get_data()
    vectorize = CountVectorizer(ngram_range=(1, 1), max_df=0.8,
                                max_features=100).fit(data['text'])
    features = vectorize.transform(data['text'])
    model = LogisticRegression(random_state=42).fit(features, data['stars'])
    return vectorize, model

# predict new input:
def predict_new(input, vector, model):
    x = vector.transform(input)
    result = model.predict(x)   
    result_p = model.predict_proba(x)
    for i, r in enumerate(result):
        dic = dict()
        dic['label'] = r
        for j in range(1, 6):
            dic[f'rating{j}_probability'] = result_p[i][j - 1]

        print(json.dumps(dic))
        print('\n')


if __name__ == '__main__':
    input = np.array([
        "This is so highly rated for a reason. \
         If you're looking for the best in Boulder for Italian, \
         go here! The food is absolutely delicious. The smell from \
         when you walk in is intoxicating. It's worth the price. \
         The portions are not huge for the price but they're the right \
         size in my opinion. The butternut squash ravioli with sage are \
         a must-try. You can tell they take a ton of pride in their food. \
         Its authentic, delicious, beautiful. It does not have a fine \
         dining atmosphere, in fact the building has a very humble \
         hole-in-the-wall feel to it, nothing like most of the other \
         Italian places in Boulder. But I actually prefer that type of \
         atmosphere anyway. Highly recommend!!",
         "Best pizza in the neighborhood!!! \
          Love the this crust, moderate amount of sauce and cheese which we like!",
         "Bagels are decent enough, but the coffee is horrible! \
          It's the worst my wife and I have ever tasted.\n\n\
          This was the Downtown Winter Garden location today."
    ])

    vector, model = train_model()
    predict_new(input, vector, model)

