import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.stats as sstats

iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Use read_csv to load the data. Make sure you get 150 examples!
iris_df = pd.read_csv(iris_url, header=None)

# Set the column names to
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']


def KNN_loo(train_X, train_Y, ks, limit=None, verbose=False):

    train_X = train_X.astype(np.float32)
    n = len(train_Y)

    if limit is not None:
        if limit <= n:
            train_X = np.array(train_X[0:limit], copy=True)
            train_Y = np.array(train_Y[0:limit], copy = True)

            n = limit
        elif limit > n:
            raise ValueError("Limit bigger than data size")


    if verbose:
            print("Computing distances... ", end='')

    dists = np.einsum('ij, ij ->i', train_X, train_X)[:, None] + np.einsum('ij, ij ->i', train_X, train_X) - 2 * train_X.dot(train_X.T)
    np.fill_diagonal(dists, np.inf)

    if verbose:
        print("Sorting... ", end='')

    # TODO: findes closest trainig points
    # Hint: use np.argsort
    closest = dists.argsort(axis=0)
    targets = train_Y[closest]

    preds = {}
    errs = {}

    if verbose:
        print("Computing predictions...", end='')

    for k in ks:
        preds[k] = np.array(sstats.mode(targets[0:k,], axis = 0)[0])
        errs[k] = (preds[k].ravel() != train_Y).mean()

    if verbose:
        print("Done")

    return preds, errs

iris_x = np.array(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])


np.random.seed(1337)
k = [15]
results = []
n = iris_x.shape[0]
for _rep in tqdm(np.arange(100)):
    train_idx = np.random.permutation(np.arange(n))
    for train_size in np.array([5, 10, 20, 50, 75, 100, 150]):
        errs = KNN_loo(iris_x[train_idx], iris_y[train_idx], k, limit=train_size)[1][15]
        results.append({'train_size':train_size, 'err_rate': errs})

# results_df will be a data_frame in long format
results_df = pd.DataFrame(results)
results_df.to_csv('Assignment3_data/prob2_2.csv', index=False)