import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

# hyperparameter tuning
def grid_search(model, x_train, y_train, grid, n_gram):

    search = RandomizedSearchCV(model, grid, random_state=42,
                                n_iter=5, cv=3, scoring='f1_micro')
    search.fit(x_train, y_train)
    result = pd.DataFrame(search.cv_results_)
    result['n_gram'] = n_gram
    print(result)
    return search

# evaluate result on hold-out test set
def evaluate(model, x_test, y_test):

    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"f1 result on test set:\n{f1}")