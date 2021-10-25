import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
import dataset_stats as ds

# evaluate model result given selected parameters
def evaluate(x_train, x_test, y_train, y_test, **kwargs):
    
    model = SVC(decision_function_shape='ovo',
                random_state=42,
                max_iter=500, **kwargs).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    return model, precision, recall, f1_score

# organize model results in a data frame
def print_result(model, ngram, precision, recall, f1_score):
    return pd.DataFrame.from_dict({
        'parameters': [model.get_params()],
        'BOW': [f'{ngram}-gram'],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })

# run experiments
def main():

    # 1-gram and 2-gram features
    x_train, x_test, y_train, y_test = ds.preprocess(1)
    x_train2, x_test2, y_train2, y_test2 = ds.preprocess(2)
    m, p, r, f = evaluate(x_train, x_test, y_train, y_test)
    base_result = print_result(m, 1, p, r, f)

    # experiment 1: 1-gram vs 2-gram with default parameters
    m1, p1, r1, f1 = evaluate(x_train2, x_test2, y_train2, y_test2)
    model1_result = print_result(m1, 2, p1, r1, f1)

    # experiment 2: 1-gram with different kernels
    m2, p2, r2, f2 = evaluate(x_train, x_test, y_train, y_test,
                              kernel='linear')
    model2_result = print_result(m2, 1, p2, r2, f2)

    # experiment 3: 1-gram with different kernels
    m3, p3, r3, f3 = evaluate(x_train, x_test, y_train, y_test,
                              kernel='poly')
    model3_result = print_result(m3, 1, p3, r3, f3)

    return pd.concat([base_result, model1_result, model2_result, model3_result]). \
                reset_index(drop=True)


if __name__ == '__main__':

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', None)
    print(main())