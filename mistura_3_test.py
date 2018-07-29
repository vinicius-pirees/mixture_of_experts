import numpy as np
import pandas
import os
from mistura_3 import mistura
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


    m = 5
    hidden_units = 7

    me = mistura(Xtr, Ytr, m, hidden_units)



    ## Teste

    Nv = Xv.shape[0]
    ne = Xv.shape[1]
    ns = Yv.shape[1]

    ##add bias
    Xv = np.concatenate((Xv, np.ones((Nv, 1))), axis=1)
    ne = ne + 1

    Wg = me['gating_Wg']
    Wg2 = me['gating_Wg2']
    W1 = me['expert_W1']
    W2 = me['expert_W2']
    var = me['expert_var']

    ##calcula saida
    Zg1 = np.dot(Xv, Wg.T)
    ##tanh
    Ag1 = (np.exp(Zg1) - np.exp(-Zg1)) / (np.exp(Zg1) + np.exp(-Zg1))
    ##add bias
    Ag1 = np.concatenate((Ag1, np.ones((Nv, 1))), axis=1)
    Zg2 = np.dot(Ag1, Wg2.T)
    Yg = softmax(Zg2)

    Ye = {}
    for i in range(m):
        Z1 = np.dot(Xv, W1[i].T)
        A1 = (np.exp(Z1) - np.exp(-Z1)) / (np.exp(Z1) + np.exp(-Z1))
        ##add bias
        A1 = np.concatenate((A1, np.ones((Nv, 1))), axis=1)
        Ye[i] = np.dot(A1, W2[i].T)
    Ym = np.zeros((Nv, ns))
    for i in range(m):
        Yge = Yg[:, i].reshape(Nv, 1)
        Ym = Ym + Ye[i] * Yge

    ##calculo da verossimilhanca
    Py = np.zeros((Nv, m))
    for i in range(m):
        Yaux = Ye[i]
        for j in range(Nv):
            diff = Yv[j, :] - Yaux[j, :]
            Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

    likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))


    print(likelihood)

    errov = Ym - Yv
    EQMv = 1 / Nv * np.sum(errov * errov)

    print(EQMv)

    print('\n',me)







