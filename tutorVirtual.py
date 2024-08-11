import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_widths
from scipy.special import softmax
import os
import pickle

with open('paramLet.pkl', 'rb') as f:
    Wl1, bl1, Wl2, bl2 = pickle.load(f)

with open('paramFon.pkl', 'rb') as f:
    Wf1, bf1, Wf2, bf2 = pickle.load(f)

F0 = pd.read_csv('ConjuntoFonemas.csv', sep='\t', header=None)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def modelFon(x, W1, b1, W2, b2):
    N = x.shape[0]
    v1 = np.tanh(W1.dot(x.transpose()) + np.outer(b1, np.ones(N)))
    v2 = W2.dot(v1) + np.outer(b2, np.ones(N))
    return softmax(v2, axis=0).transpose()

def modelLet(x, W1, b1, W2, b2):
    N = x.shape[0]
    v1 = np.tanh(W1.dot(x.transpose()) + np.outer(b1, np.ones(N)))
    return sigmoid(W2.dot(v1) + np.outer(b2, np.ones(N))).flatten()

def LPC3janelas(s, fa):
    N = s.shape[0]
    if (N >= int(np.round(0.02*fa))) & (N <= int(np.round(0.2*fa))):
        Janela = int(np.round(N/3))
        marcas = [0, int(np.round((N-Janela)/2))-1, N-Janela]
        Ordem = int(np.round(0.003*fa))
        CP = np.zeros((Ordem,3))
        S = np.zeros((Janela-Ordem-1,Ordem+1))
        cont = 0
        for k in marcas:
            saux = s[k:k+Janela]
            for m in range(Ordem+1):
                S[:,m] = saux[m:m-Ordem-1]
            C = np.linalg.pinv(S[:,:-1]).dot(S[:,-1])
            CP[:,cont] = C
            cont += 1
        return CP
    else:
        return []
    
def CP2vec(CP, fa):
    w = np.arange(5000/fa*np.pi, 50/fa*np.pi, -100/fa*np.pi)
    P = np.zeros((len(w), CP.shape[1]))
    for k in range(CP.shape[1]):
        h = np.hstack((1, -np.flipud(CP[:,k])))
        for i in range(len(w)):
            P[i,k] = abs(1/np.sum(np.exp(-1j*w[i]*np.arange(len(h)))*h))
    P = np.log10(P/P.max() + 0.01) + 2
    P = P.flatten(order='F')/np.sqrt(np.sum(P.flatten()**2))
    return P

def lempelziv76(s):
	K = len(np.unique(s))
	N = len(s)
	L = 1
	dic = [s[0]]
	p = 1
	L = L+1
	while p+L < N:
		pos = ''.join(s[:p+L-1]).find(''.join(s[p:p+L]))
		if pos == -1:
			dic.append(''.join(s[p:p+L]))
			p = p+L
			L = 1
		else:
			L = L+1
	dic.append(''.join(s[p:]))
	cLZ = len(dic)*(np.log2(len(dic))+1)/N
	return dic, cLZ

def perfEner(s, fa):
    N = s.shape[0]
    janela = np.round(0.1*fa).astype(int)
    passo = np.round(0.03*fa).astype(int)
    N2 = (N-janela)//passo
    E = np.zeros(N2)
    for i in range(N2):
        saux = s[i*passo:i*passo+janela]
        E[i] = (saux**2).sum()
    return E

def segmenta(E, pfala):
    segs = np.empty((2,0), dtype=int)
    qp = np.zeros(pfala.shape[1])
    for i in range(pfala.shape[1]):
        aux = E[pfala[0,i]:pfala[1,i]]
        peaks, _ = find_peaks(aux, distance=4)
        if peaks.size == 0: continue
        proe = signal.peak_prominences(aux/max(aux), peaks)[0]
        if np.any(proe/max(proe)<0.01): peaks = peaks[proe>0.01]
        _, _, ini, fim = peak_widths(aux, peaks, rel_height=0.7)
        ini = np.round(ini).astype(int)
        fim = np.round(fim).astype(int)
        qp[i] = len(peaks)
        segs = np.hstack((segs, np.stack((ini,fim)) + pfala[0,i]))
    return segs, qp

def filtrarSilencios(s, fa):
    E = perfEner(s, fa)
    passo = np.round(0.03*fa).astype(int)
    pz = np.logical_and(E[1:] > E.max()/100, E[:-1] < E.max()/100)
    nz = np.logical_and(E[1:] < E.max()/100, E[:-1] > E.max()/100)
    pz = np.nonzero(pz)[0]
    nz = np.nonzero(nz)[0] + 1
    if nz[0]<=pz[0]: nz = nz[1:]
    if nz[-1]<=pz[-1]: pz = pz[:-1]
    fmed = (nz-pz).mean()
    pausas = pz[1:] - nz[:-1]
    flags = np.ones(len(s), dtype=bool)
    for i in np.nonzero(pausas>fmed*2)[0]:
        ini = (nz[i]+2*int(fmed))*passo
        fin = int(pz[i+1])*passo
        flags[ini:fin] = False
    s = s[flags]
    return s

def wav2ener2fon2(s, fa):
    E = perfEner(s, fa)
    pz = np.logical_and(E[1:] > E.max()/100, E[:-1] < E.max()/100)
    nz = np.logical_and(E[1:] < E.max()/100, E[:-1] > E.max()/100)
    pz = np.nonzero(pz)[0]
    nz = np.nonzero(nz)[0] + 1
    if nz[0]<=pz[0]: nz = nz[1:]
    if nz[-1]<=pz[-1]: pz = pz[:-1]
    pfala = np.stack((pz,nz))
    segs, _ = segmenta(E, pfala)
    P2 = np.zeros((segs.shape[1], 150))
    for i in range(segs.shape[1]):
        na = np.round(segs[0,i]*fa*0.03).astype(int)
        N = np.round((segs[1,i]-segs[0,i])*fa*0.03).astype(int)
        janela = int(np.round(N/3))
        marcas = [0, int(np.round((N-janela)/2))-1, N-janela]
        Ordem = int(np.round(0.003*fa))
        CP = np.zeros((Ordem,3))
        S = np.zeros((janela-Ordem-1,Ordem+1))
        cont = 0
        for k in marcas:
            saux = s[na+k:na+k+janela]
            for m in range(Ordem+1):
                S[:,m] = saux[m:m-Ordem-1]
            C = np.linalg.pinv(S[:,:-1]).dot(S[:,-1])
            CP[:,cont] = C
            cont += 1
        P2[i,] = CP2vec(CP, fa).transpose()
    yp = modelFon(P2, Wf1, bf1, Wf2, bf2)
    inds = yp.argmax(axis=1)
    aux = F0.values[inds].flatten()
    fonemas = np.zeros(E.shape[0], dtype=np.str_)
    fonemas[:] = '0'
    for i in range(segs.shape[1]):
        fonemas[segs[0,i]:segs[1,i]] = aux[i]
    return fonemas

def taxaLetras(s, fa):
    E = perfEner(s, fa)
    pz = np.logical_and(E[1:] > E.max()/100, E[:-1] < E.max()/100)
    nz = np.logical_and(E[1:] < E.max()/100, E[:-1] > E.max()/100)
    pz = np.nonzero(pz)[0]
    nz = np.nonzero(nz)[0] + 1
    if nz[0]<=pz[0]: nz = nz[1:]
    if nz[-1]<=pz[-1]: pz = pz[:-1]
    pfala = np.stack((pz,nz))
    P2 = np.zeros((pfala.shape[1], 150))
    for i in range(pfala.shape[1]):
        na = np.round(pfala[0,i]*fa*0.03).astype(int)
        N = np.round((pfala[1,i]-pfala[0,i])*fa*0.03).astype(int)
        janela = int(np.round(N/3))
        marcas = [0, int(np.round((N-janela)/2))-1, N-janela]
        Ordem = int(np.round(0.003*fa))
        CP = np.zeros((Ordem,3))
        S = np.zeros((janela-Ordem-1,Ordem+1))
        cont = 0
        for k in marcas:
            saux = s[na+k:na+k+janela]
            for m in range(Ordem+1):
                S[:,m] = saux[m:m-Ordem-1]
            C = np.linalg.pinv(S[:,:-1]).dot(S[:,-1])
            CP[:,cont] = C
            cont += 1
        P2[i,] = CP2vec(CP, fa).transpose()
    yp = modelLet(P2, Wl1, bl1, Wl2, bl2)
    return (yp>0.5).sum()/len(yp)