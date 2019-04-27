import random
import numpy as np
from numpy import linalg as LA

def Dist_CX(C,X,c,n):
    D = np.zeros((c,n),dtype=np.float64) #matriz D das distâncias para atualização de U
    for i in range(c):
        for j in range(n):
            D[i,j] = LA.norm(C[i]-X[j])
    return D

def Atual_U(U,D,e,c,n):
    for i in range(c):
        for j in range(n):
            s = 0
            for l in range(c):
                if D[i,j] == 0:
                    if l != i:
                        U[l,j] = 0
                        U[i,j] = 1
                else:
                    s = s + D[l,j]**e
                    U[i,j] = (D[i,j]**e)/s
    return U

def Atual_C(C,U,X,mu,c,n):
    for i in range(c):
        sx = 0
        s = 0
        for j in range(n):
            sx = sx + ((U[i,j])**mu)*X[j]
            s = s + U[i,j]**mu           
        C[i] = sx/s
    return C

def C_init(X,c):
    [n,m] = X.shape
    ii = random.sample(range(n),c) # tomando c elementos aleatórios de X
    # iniciando a matriz dos centroides aleatória.
    C = np.zeros((c,m))
    for i in range(len(ii)):
        C[i] = X[ii[i],:] # C é a matriz cujas linhas são os centroides
#     C1 = C # Guarda matriz dos centroides iniciais
    return C

def FCM(X,mu,c,eps,itmax,C):
    [n,m] = X.shape  
    # iniciando a matriz U = (Uij) de pertinencia do xj pertencer ao cluste ci
    U = np.random.rand(c,n) # U não precisa ser inicializada. Ela será atualizada ao decorrer do algoritmo
    d = 1
    e = 2/(1-mu)
    it = 1
    while d > eps and it <= itmax:
        # print("iteração ",it)
        C2 = np.zeros((c,m),dtype=np.float64)  # matriz de centroides para atualização
        D = Dist_CX(C,X,c,n)
        U = Atual_U(U,D,e,c,n)
        C2 = Atual_C(C2,U,X,m,c,n)
        V = C-C2
        d = LA.norm(V) # diferença entre a matriz de centroides anterior com a nova   
        C = C2
        it = it + 1    
        
    return (C,U,it)