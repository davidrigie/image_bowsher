import numpy as np
import ipdb
import numba
from numba import int_ as int_
from numba import float64 as float_
from scipy.sparse.linalg import LinearOperator    

@numba.njit(int_(int_,int_))
def min(x1,x2):
    if x2 < x1:
        return x2
    return x1

@numba.njit(int_(int_,int_))
def max(x1,x2):
    if x1 > x2:
        return x1

    return x2

@numba.njit(int_(int_))
def factorial(n):

    y = 1
    for i in range(n):
        y *= (i+1)

    return y

@numba.njit(int_(int_,int_))
def nchoosek(n,k):

    numer = factorial(n)
    denom = factorial(k)*factorial(n-k)

    return numer/denom


@numba.jit(int_[:,:,:](float_[:,:]))
def generate_graph(x):
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    neighborhood_size = 4
    search_size = 8

    g = np.zeros((n_rows, n_cols, neighborhood_size), dtype=np.float)
    
    w_sz = int(search_size/2)

    for i in range(n_rows):
        for j in range(n_cols):
            istart = max(0, i-w_sz)
            istop  = min(n_rows, i+w_sz)
            jstart = max(0, j-w_sz)
            jstop  = min(n_cols, j+w_sz)

            buffer = x[istart:istop, jstart:jstop] - x[i,j]
            ind = np.argsort(buffer.reshape(-1))
            ind += (istart*n_cols + jstart)

            g[i,j,:] = ind[0:4]

    return g.astype(int)
    
@numba.njit
def graph_diff(x, g):

    n_rows = g.shape[0]
    n_cols = g.shape[1]
    neighborhood_size = g.shape[2]

    output_channels = nchoosek(neighborhood_size, 2)
    y = np.zeros((n_rows, n_cols, output_channels))

    x1d = x.reshape(-1)

    for i in range(n_rows):
        for j in range(n_cols):
            dim_counter = 0
            for k1 in range(neighborhood_size):
                for k2 in range(k1+1, neighborhood_size):
                    y[i,j,dim_counter] = x1d[g[i,j,k1]] - x1d[g[i,j,k2]]
                    dim_counter += 1

    return y

@numba.njit
def graph_diff_adj(x, g):

    n_rows = g.shape[0]
    n_cols = g.shape[1]
    neighborhood_size = g.shape[2]

    y = np.zeros((n_rows, n_cols))

    y1d = y.reshape(-1)

    for i in range(n_rows):
        for j in range(n_cols):
            dim_counter = 0
            for k1 in range(neighborhood_size):
                for k2 in range(k1+1, neighborhood_size):
                    ind1 = g[i,j,k1] 
                    ind2 = g[i,j,k2] 
                    y1d[ind1] += x[i,j,dim_counter]
                    y1d[ind2] += -x[i,j,dim_counter]                    
                    dim_counter += 1
    return y
                    

def generate_linop(f, fadj, x0):

    y0 = f(x0)
    shape = (y0.size, x0.size)
    
    class MyLinop(LinearOperator):

        def __init__(self):
            super().__init__(None, shape)

        def _matvec(self, v):
            v2d = v.reshape(x0.shape)
            val = f(v2d).ravel()
            return val

        def _rmatvec(self, v):
            v2d = v.reshape(y0.shape)
            val = fadj(v2d).ravel()
            return val

        def _transpose(self):
            return self._adjoint()

        def test_adjoint(self, N=10):
            v1 = np.zeros(N)
            v2 = v1*0.0
            for i in range(N):
                x = np.random.random(self.shape[1])
                y = np.random.random(self.shape[0])
                val1 = self.dot(x).dot(y)
                vaL2 = x.dot(self.rmatvec(y))
                print('<Kx,y> = {}\n<x,K^Hy> = {}'.format(val1,vaL2))
                v1[i] = val1
                v2[i] = vaL2
            
            if not np.allclose(v1,v2):
                print('Failed the adjoint test!')
                return False
            
            return True

        def norm(self, N=10):
            x = np.random.random(self.shape[1])
            for i in range(N):
                Kx = self.dot(x)
                s = np.linalg.norm(Kx)
                x = self.rmatvec(Kx)
                x /= np.linalg.norm(x)
                print('Spectral Norm: {}'.format(s))
            
            return s
                

    A = MyLinop()

    return A

def shrink(x):

    val = x/np.maximum(1, np.abs(x))
    return val

def cp_L2_kl1_iter(x, y, K, beta):

    # init
    s = K.norm(100)
    gamma = 1.0/beta
    tau = 1/s
    sigma = 1/s
    z = K.dot(x)
    xbar = x*1.0

    while True:
        # dual-update
        z = shrink(z + sigma*K.dot(xbar))

        # primal-update
        x_prev = x*1.0
        x = (tau*y + beta*x)/(tau + beta)
        
        # choose step-sizes
        theta = 1.0/np.sqrt(1+2*gamma*tau)
        tau *= theta
        sigma /= theta

        # extrapolation step
        xbar = x + theta*(x-x_prev)

        yield x
    
def cp_L2_kl1(x, y, K, beta, maxiter=1000):

    generator = cp_L2_kl1(x, y, K, beta)

    for i in range(maxiter):
        x = next(generator)

    return x

if __name__ == '__main__':

    N = 512
    x = np.random.random([N,N])
    g = generate_graph(x)

    f    = lambda x : graph_diff(x,g)
    fadj = lambda x : graph_diff_adj(x,g)

    K = generate_linop(f, fadj, x)


    





            



                           
