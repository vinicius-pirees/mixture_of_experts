import numpy as np
import pandas
import os
from utils import series_to_supervised


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

        folds[i+1] = Dataset[_from:_to, :]

    return folds


def k_fold(D,k):
    folds = split_dataset(D, k)

    folded_partitions = {}

    for fold in folds:
        train_folds = [folds[x] for x in range(1, k + 1) if x != fold]
        Dtr = np.concatenate(train_folds)
        Dv = folds[fold]

        folded_partitions[fold] = [Dtr, Dv]

    return folded_partitions


if __name__ == '__main__':

    filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
    series = pandas.read_csv(filename, header=None)

    D = series_to_supervised(series, 15).values
    k = 4

    folded_partitions = k_fold(D,k)

    for partition in folded_partitions:
        Dtr = folded_partitions[partition][0]
        Dv = folded_partitions[partition][1]

        Xtr = Dtr[:, 0:-1]
        Ytr = Dtr[:, -1].reshape(Xtr.shape[0], 1)

        Xv = Dv[:, 0:-1]
        Yv = Dv[:, -1].reshape(Xv.shape[0], 1)

        print("Xtr", partition, ": ", Xtr)
        print("Xv", partition, ": ", Xv)
        print("")
