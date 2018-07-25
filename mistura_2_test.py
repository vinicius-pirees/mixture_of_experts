import numpy as np
import pandas
import os
from mistura_2 import mistura, softmax
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


if __name__ == "__main__":

    D = series_to_supervised(series, 21).values

    X = D[:,0:-1]
    Y = D[:,-1].reshape(X.shape[0],1)

    train_size = round(X.shape[0] * 0.7) # Dataset de treinamento terá 70% dos dados
    test_size = X.shape[0] - train_size
    Xtr = X[0:train_size,:]
    Xv = X[train_size:train_size+test_size,:]
    Ytr = Y[0:train_size,:]
    Yv = Y[train_size:train_size+test_size,:]


    m = 4
    hidden_units = 7

    me = mistura(Xtr, Ytr, m, hidden_units)



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

    print('\n',me)







