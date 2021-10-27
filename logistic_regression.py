import pandas as pd
from sklearn.linear_model import LogisticRegression
import dataset_stats as ds
from train_test import grid_search, evaluate

# run experiments
def main():

    # 1-gram and 2-gram features
    x_train, x_test, y_train, y_test = ds.preprocess(1)
    x_train2, x_test2, y_train2, y_test2 = ds.preprocess(2)

    # hyperparameters to tune
    param_dist = {
        'C': [0.8, 0.9, 1.0],
        'solver': ['lbfgs', 'newton-cg', 'saga']
    }

    model = LogisticRegression(random_state=42, max_iter=80)
    
    # experiments: 1-gram with grid search
    print('experiment1: 1-gram with grid search')
    model1 = grid_search(model, x_train, y_train, param_dist, 1)
    evaluate(model1, x_test, y_test)
    print('\n')

    # experiment 2: 2-gram with grid search
    print('experiment2: 2-gram with grid search')
    model2 = grid_search(model, x_train2, y_train2, param_dist, 2)
    evaluate(model2, x_test2, y_test2)


if __name__ == '__main__':

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', None)
    main()