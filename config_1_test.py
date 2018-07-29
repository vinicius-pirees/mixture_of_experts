import numpy as np
import pandas
import os
from config_1_perceptron import perceptron, calc_saida
from utils import series_to_supervised

filename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname('treinamento.txt'))) + '/treinamento-1.txt'
series = pandas.read_csv(filename,  header=None)


if __name__ == "__main__":

    D = series_to_supervised(series, 5).values

    X = D[:,0:-1]
    Y = D[:,-1].reshape(X.shape[0],1)

    train_size = round(X.shape[0] * 0.7) # Dataset de treinamento terá 70% dos dados
    test_size = X.shape[0] - train_size
    Xtr = X[0:train_size,:]
    Xv = X[train_size:train_size+test_size,:]
    Ytr = Y[0:train_size,:]
    Yv = Y[train_size:train_size+test_size,:]

    W, vet_erro = perceptron(50000, 0.1, Xtr, Ytr)

    ## Teste

    Nv = Yv.shape[0]
    Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)

    ##Calcula saida para conjunto de teste
    Av, errov = calc_saida(Xv, Yv, W)
    EQMv = 1 / Nv * np.sum(errov * errov)

    print(EQMv)







