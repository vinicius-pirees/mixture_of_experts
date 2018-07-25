import numpy as np
import pandas
import os


filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
series = pandas.read_csv(filename,  header=None)


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




D = series_to_supervised(series, 21).values

k = 4

print(D.shape)

def split_dataset(Dataset, k):
    N = Dataset.shape[0]
    fold_size = int(N/k)

    folds = {}

    for i in range(k):
        _from = i * fold_size

        if(i+1 == k):
            _to = N
        else:
            _to = (i+1) * fold_size

        folds[i+1] = D[_from:_to, :]

    return folds

partitions = split_dataset(D, k)

for key in partitions:
    train_folds = [partitions[x] for x in range(1,k+1) if x != key]
    Dtr = np.concatenate(train_folds)
    Dv = partitions[key]

    print("Treinamento ", key)
    print(Dtr.shape)
    print(Dtr)

    print("Validação ", key)
    print(Dv.shape)
    print(Dv)

    Xtr = Dtr[:, 0:-1]
    Ytr = Dtr[:, -1].reshape(Xtr.shape[0], 1)

    Xv = Dv[:, 0:-1]
    Yv = Dv[:, -1].reshape(Xv.shape[0], 1)

