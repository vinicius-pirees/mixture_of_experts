import numpy as np
import pandas
import os
from config_3_mistura_perceptron import mistura
from utils import series_to_supervised, softmax

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

    m = 7

    me = mistura(Xtr, Ytr, m)



    ## Teste

    Nv = Xv.shape[0]
    ne = Xv.shape[1]
    ns = Yv.shape[1]

    ##add bias
    Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)
    ne = ne + 1

    Wg = me['gating']
    W = me['expert_W']
    var = me['expert_var']

    ##calcula a saida
    Yg = softmax(np.dot(Xv, Wg.T))
    Ye = {}
    for i in range(m):
        Ye[i] = np.dot(Xv, W[i].T)
    Ym = np.zeros((Nv, ns))
    for i in range(m):
        Yge = Yg[:, i].reshape(Nv, 1)
        Ym = Ym + Ye[i] * Yge

    ##calculo da funcao de verossimilhanca
    Py = np.zeros((Nv, m))
    for i in range(m):
        Yaux = Ye[i]
        for j in range(Nv):
            diff = Yv[j, :] - Yaux[j, :]
            Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

    likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))

    erro_medio = np.sqrt(np.square(np.sum(Yv - Ym))/Nv)

    print (likelihood)

    errov = Ym - Yv
    EQMv = 1 / Nv * np.sum(errov * errov)

    print(EQMv)

    print('\n',me)







