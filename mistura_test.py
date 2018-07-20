import numpy as np
import pandas
import os

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

def mistura(Xtr, Ytr, m):
    Ntr = Xtr.shape[0]
    ne = Xtr.shape[1]
    ns = Ytr.shape[1]

    ##add bias
    Xtr = np.concatenate((Xtr, np.ones((Ntr, 1))), axis=1)
    ne = ne + 1

    ## Inicializa rede gating
    Wg = np.random.rand(m, ne)

    # Inicializa especialistas
    W = {}
    var = list(range(m))
    for i in range(m):
        W[i] = np.random.rand(ns, ne)
        var[i] = 1

    ##calcula saida
    Yg = softmax(np.dot(Xtr, Wg.T))

    Ye = {}
    for i in range(m):
        Ye[i] = np.dot(Xtr, W[i].T)
    Ym = np.zeros((Ntr, ns))
    for i in range(m):
        Yge = Yg[:, i].reshape(Ntr, 1)
        Ym = Ym + Ye[i] * Yge

    ##calculo da verossimilhanca
    Py = np.zeros((Ntr, m))
    for i in range(m):
        Yaux = Ye[i]
        for j in range(Ntr):
            diff = Ytr[j, :] - Yaux[j, :]
            Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

    likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))
    likelihood_ant = 0
    nit = 0
    nitmax = 100

    while np.abs(likelihood - likelihood_ant) > 1e-3 and nit < nitmax:
        nit = nit + 1
        # Passo E
        haux = Yg * Py
        h = haux / np.dot(np.sum(haux, axis=1, keepdims=True), np.ones((1, m)))
        ##Passo M
        Wg = maximiza_gating(Wg, Xtr, m, h)
        for i in range(m):
            W[i], var[i] = maximiza_expert(W[i], var[i], Xtr, Ytr, h[:, i].reshape(Ntr, 1))
        likehood_ant = likelihood

        ##calcula a saida
        Yg = softmax(np.dot(Xtr, Wg.T))
        Ye = {}
        for i in range(m):
            Ye[i] = np.dot(Xtr, W[i].T)
        Ym = np.zeros((Ntr, ns))
        for i in range(m):
            Yge = Yg[:, i].reshape(Ntr, 1)
            Ym = Ym + Ye[i] * Yge

        ##calculo da funcao de verossimilhanca
        Py = np.zeros((Ntr, m))
        for i in range(m):
            Yaux = Ye[i]
            for j in range(Ntr):
                diff = Ytr[j, :] - Yaux[j, :]
                Py[j, i] = np.exp(np.dot(-diff, diff.T) / (2 * var[i]))

        likelihood = np.sum(np.log(np.sum(Yg * Py, axis=1, keepdims=True)))

    me = {}
    me['gating'] = Wg
    me['expert_W'] = W
    me['expert_var'] = var

    return me

def softmax(Z):
    Z_exp = np.exp(Z)
    Z_sum = np.sum(Z_exp, axis = 1, keepdims = True)
    return Z_exp/Z_sum


def maximiza_gating(Wg, Xtr, m, h):
    N = Xtr.shape[0]
    ne = Xtr.shape[1]
    Yg = softmax(np.dot(Xtr, Wg.T))

    grad = np.dot((h - Yg).T, (Xtr / N))
    dir = grad
    nit = 0
    nitmax = 50000
    alfa = 0.1

    while np.linalg.norm(grad) > 1e-5 and nit < nitmax:
        nit = nit + 1
        Wg = Wg + (alfa * dir)
        Yg = softmax(np.dot(Xtr, Wg.T))
        grad = np.dot((h - Yg).T, (Xtr / N))
        dir = grad

    return Wg


def maximiza_expert(W, var, Xtr, Ytr, h):
    Ye = np.dot(Xtr, W.T)
    N = Ye.shape[0]
    ns = Ye.shape[1]

    grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)

    dir = grad
    nit = 0
    nitmax = 50000
    alfa = 0.1

    while np.linalg.norm(grad) > 1e-5 and nit < nitmax:
        nit = nit + 1
        W = W + (alfa * dir)
        Ye = np.dot(Xtr, W.T)
        grad = np.dot(((h / var) * (Ytr - Ye)).T, Xtr / N)
        dir = grad

    diff = Ytr - Ye
    soma = 0

    for i in range(m):
        soma = soma + (h[i] * np.dot(diff[i, :], diff[i, :].T))
    var = max(0.05, (1 / ns) * soma / np.sum(h))

    return W, var

def series_to_supervised(df, n_lags, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a pandas dataframe.
        n_lags: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_lags, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pandas.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == "__main__":

    D = series_to_supervised(series, 3).values

    X = D[:,0:-1]
    Y = D[:,-1].reshape(X.shape[0],1)

    train_size = round(X.shape[0] * 0.7) # Dataset de treinamento terá 70% dos dados
    test_size = X.shape[0] - train_size
    Xtr = X[0:train_size,:]
    Xv = X[train_size:train_size+test_size,:]
    Ytr = Y[0:train_size,:]
    Yv = Y[train_size:train_size+test_size,:]


    m = 4

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

    ##calcula a saida
    Yg = softmax(np.dot(Xv, Wg.T))
    Ye = {}
    for i in range(m):
        Ye[i] = np.dot(Xv, W[i].T)
    Ym = np.zeros((Nv, ns))
    for i in range(m):
        Yge = Yg[:, i].reshape(Nv, 1)
        Ym = Ym + Ye[i] * Yge

    erro_medio = np.sqrt(np.square(np.sum(Yv - Ym))/Nv)

    print (erro_medio)







