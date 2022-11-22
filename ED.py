# Exact diagonalization for spin chain
# 2017 Aug. Tsuyoshi Okubo
# 2018 Dec. updated to use matrix-vector multiplication

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as spr
import scipy.sparse.linalg as spr_linalg
import argparse

class Hamiltonian:
    def __init__(self,m,Jz,Jxy,hx,D,N,periodic=False):
        self.Jz = Jz
        self.Jxy = Jxy
        self.hx = hx
        self.D = D
        self.N = N
        self.periodic = periodic
        self.m = m
        v_shape = (m,)
        for i in range(1,N):
            v_shape += (m,)
        self.v_shape = v_shape
        
        Sp = np.zeros((m,m))
        for i in range(1,m):
            Sp[i-1,i] = np.sqrt(i * (m - i))

        Sm = np.zeros((m,m))
        for i in range(0,m-1):
            Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

        Sz = np.zeros((m,m))
        for i in range(m):
            Sz[i,i] = 0.5 * (m - 1.0) - i
        
        
        Id = np.identity(m)
        Sx = 0.5 * (Sp + Sm)
        Sz2 = np.dot(Sz,Sz)


        if self.periodic:
            self.pair_operator = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) + 0.5 * D * (np.kron(Sz2,Id)
                                                                                              + np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator = self.pair_operator
        else:
            self.pair_operator = (Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) +np.kron(Sm,Sp))
                                  - 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) + 0.5 * D * (np.kron(Sz2,Id)
                                                                                              + np.kron(Id,Sz2))).reshape(m,m,m,m)
            self.periodic_pair_operator = (- 0.5 * hx * (np.kron(Sx,Id) + np.kron(Id,Sx)) + 0.5 * D * (np.kron(Sz2,Id)
                                                                                              + np.kron(Id,Sz2))).reshape(m,m,m,m)
            
                            
        ## shape of transepose after applying pair operators
        self.pair_transpose_list = []
        for i in range(0,N-1):
            self.pair_transpose_list.append(tuple(np.arange(0,i,dtype=int)) + (N-2,N-1) + tuple(np.arange(i,N-2,dtype=int)))
        ## for N-1 (periodic boundary)
        self.pair_transpose_list.append((N-1,) + tuple(np.arange(0,N-2,dtype=int)) + (N-2,))

    def mult_Hamiltonian(self,v):
        x = np.zeros(self.v_shape)
        vr = v.reshape(self.v_shape)
        
        for i in range(self.N - 1):
            x += np.tensordot(vr,self.pair_operator,axes=([i,i+1],[2,3])).transpose(self.pair_transpose_list[i])
        x += np.tensordot(vr,self.periodic_pair_operator,axes=([self.N-1,0],[2,3])).transpose(self.pair_transpose_list[self.N-1])                     
        return x.reshape(self.m**self.N)
           
def Calc_GS(m,Jz,Jxy,hx,D,N,k=5,periodic=False):
    hamiltonian = Hamiltonian(m,Jz,Jxy,hx,D,N,periodic)
    Ham = spr_linalg.LinearOperator((m**N,m**N),hamiltonian.mult_Hamiltonian,dtype=float)
    eig_val,eig_vec = spr_linalg.eigsh(Ham,k=k,which="SA")
    return eig_val,eig_vec
    
#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='ED siumulator for one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N  (default = 10)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
                        help='interaction for SzSz  (default = 1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
                        help='interaction for SxSx + SySy  (default = 1.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=3,
                        help='Spin size m=2S +1  (default = 3)')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
                        help='extarnal magnetix field  (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('-e_num', metavar='e_num',dest='e_num', type=int, default=5,
                        help='number of calculating energies (default = 5)')
    parser.add_argument('--periodic',dest='periodic', action="store_true",
                        help='set periodic boundary condision (default = open)')
    return parser.parse_args()



if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    eig_val,eig_vec = Calc_GS(args.m,args.Jz,args.Jxy,args.hx,args.D,args.N,args.e_num,args.periodic)
    N = args.N
    print("S=1 N-site open Heisenberg chain")
    print("N = "+ repr(N))
    print("Ground state energy per bond = "+ repr(eig_val[0]/(N-1)))
    for i in range(1,args.e_num):
        print("Excited states " +repr(i) +":  " + repr(eig_val[i]/(N-1)))
