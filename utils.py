import pandas
import numpy as np

def softmax(Z):
    Z_exp = np.exp(Z)
    Z_sum = np.sum(Z_exp, axis = 1, keepdims = True)
    return Z_exp/Z_sum

def dirac_delta(i,j):
    if i == j:
        return 1
    else:
        return 0

def series_to_supervised(df, n_lags, n_out=1, dropnan=True):
    """
    Converte uma série temporal para um dataset de aprendizado supervisionado.
    Arguments:
        df: Serie temporal.
        n_lags: Numero de lags (X).
        n_out: Numero de saidas (y).
        dropnan: Remover as linhas superiores com valores Nan.
    Returns:
        Dataframe pandas.
    """
    n_vars = df.shape[1]
    cols, names = list(), list()

    for i in range(n_lags, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = pandas.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg