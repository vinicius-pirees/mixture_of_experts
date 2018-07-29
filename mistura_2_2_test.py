import numpy as np
import pandas
import os
from mistura_2_2 import mistura
from utils import softmax, series_to_supervised

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%  Mistura de Especialista  %%%%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Especialista - Perceptron%%%%%%%%%%%%%%%%%%%
# %%%%%Rede Gating - Perceptron%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Xtr - entrada de treinamento
# Ytr - saida de treinamento
# Wg - rede gating
# W - especialistas

##Especialistas MLP 1 camada oculta
##Gating - perceptron

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


    m = 6

    me = mistura(Xtr, Ytr, m)

    print('\n', me, '\n')

    ## Teste

    Nv = Xv.shape[0]
    ne = Xv.shape[1]
    ns = Yv.shape[1]

    ##add bias
    Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)
    ne = ne + 1

    Wg = me['gating']
    W1 = me['expert_W1']
    W2 = me['expert_W2']
    var = me['expert_var']

    ##calcula saida
    Yg = softmax(np.dot(Xv, Wg.T))

    Ye = {}
    for i in range(4):
        Z1 = np.dot(Xv, W1[i].T)
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1)+np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((Nv, 1))), axis=1)
        Ye[i] = np.dot(A1, W2[i].T)

    for i in range(4,6):
        Ye[i] = np.dot(Xv, W1[i].T)


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

    print(likelihood)

    errov = Ym - Yv
    EQMv = 1 / Nv * np.sum(errov * errov)

    print(EQMv)









