import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse

def set_Hamiltonian_S(m,Jz,Jxy,hx,D,position=0):
    ## m = 2S+1    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    if position == 0: ##center
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - 0.5 * hx * (np.kron(Id,Sx) + np.kron(Sx,Id)) + 0.5 * D * (np.kron(Id,Sz2) + np.kron(Sz2,Id))
    elif position < 0: ## left boundary
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (0.5 * np.kron(Id,Sx) + np.kron(Sx,Id)) + D * (0.5 * np.kron(Id,Sz2) + np.kron(Sz2,Id))
    else: ## right boundary
        return Jz * np.kron(Sz,Sz) + 0.5 * Jxy * (np.kron(Sp,Sm) + np.kron(Sm,Sp)) - hx * (np.kron(Id,Sx) + 0.5 * np.kron(Sx,Id)) + D * (np.kron(Id,Sz2) + 0.5 * np.kron(Sz2,Id))

def mult_left(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),Tn_i.conj(),([0,1],[1,0]))

def mult_right(v,lam_i,Tn_i):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,2)),Tn_i.conj(),([0,1],[2,0]))

def mult_left_op(v,lam_i,Tn_i,op):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),op,(1,1)),Tn_i.conj(),([0,2],[1,0]))

def TEBD_update(Tn,lam,expH,chi_max,inv_precision=1e-10):
    N = len(Tn)
    m = expH[0].shape[0]

    lam_inv =[]

    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    for eo in range(2):
        for i in range(eo,N-1,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[i+1].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[i+1]),(2,0))
                        ,Tn[i+1],(2,1))
                    ,np.diag(lam[i+2]),(3,0))
                ,expH[i],([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[i+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[i+1] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[i+2]),(1,0)).transpose(1,0,2)

            lam_inv[i+1] = 1.0/lam[i+1]

    return Tn,lam

def TEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=1e-10):
    N = len(Tn)
    m = expH[0].shape[0]

    lam_inv =[]

    expH_eo=[expH2,expH]
    
    for i in range(N+1):
        lam_inv.append(1.0/lam[i])

    for eoe in range(3):
        if eoe == 1:
            eo = 1
        else:
            eo = 0    
        for i in range(eo,N-1,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[i+1].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[i+1]),(2,0))
                        ,Tn[i+1],(2,1))
                    ,np.diag(lam[i+2]),(3,0))
                ,expH_eo[eo][i],([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(m*chi_l,m*chi_r)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[i+1] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[i+1] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[i+2]),(1,0)).transpose(1,0,2)

            lam_inv[i+1] = 1.0/lam[i+1]

    return Tn, lam

def Calc_Environment(Tn,lam,canonical=False):
    ## Calculate left and right contraction exactly

    N = len(Tn)
    Env_left = []
    Env_right = []
    Env_right_temp = []

    if canonical:
        ## assume MPS is in canonical form
        for i in range(N):
            Env_left.append(np.identity((lam[i].shape[0])))
            Env_right.append(np.dot(np.dot(np.diag(lam[i+1]),np.identity((lam[i+1].shape[0]))),np.diag(lam[i+1])))
    else:
        left_env = np.identity(1).reshape(1,1)
        right_env = np.identity(1).reshape(1,1)
        Env_left.append(left_env)
        Env_right_temp.append(np.dot(np.dot(np.diag(lam[N]),right_env),np.diag(lam[N])))
        for i in range(1,N):

            left_env = mult_left(left_env,lam[i-1],Tn[i-1])
            right_env = mult_right(right_env,lam[N-i+1],Tn[N-i])

            Env_left.append(left_env)
            Env_right_temp.append(np.dot(np.dot(np.diag(lam[N-i]),right_env),np.diag(lam[N-i])))

        for i in range(N):
            Env_right.append(Env_right_temp[N-i-1])

    return Env_left,Env_right


def Contract_one_site(El,Er,T_i,lam_i,op):
    return np.tensordot(mult_left_op(El,lam_i,T_i,op),Er,([0,1],[0,1]))
def Contract_one_site_no_op(El,Er,T_i,lam_i):
    return np.tensordot(mult_left(El,lam_i,T_i),Er,([0,1],[0,1]))

def Contract_two_site(El,Er,T1,T2,lam1,lam2,op1,op2):
    return np.tensordot(mult_left_op(mult_left_op(El,lam1,T1,op1),lam2,T2,op2),Er,([0,1],[0,1]))
def Contract_two_site_no_op(El,Er,T1,T2,lam1,lam2):
    return np.tensordot(mult_left(mult_left(El,lam1,T1),lam2,T2),Er,([0,1],[0,1]))

def Contract_correlation(Env_left,Env_right,Tn,lam,op1,op2,max_distance,step=1):
    ## Output sequence of correlation <op1(0) op2(r)> for r <= max_distance.
    ## r is increased by step

    N = len(Tn)
    Correlation=[]
    El = Env_left[0]
    Er = Env_right[0]
    
    El_op = mult_left_op(El,lam[0],Tn[0],op1)
    El_identity = mult_left(El,lam[0],Tn[0])
    for j in range(1,step):
        El_op = mult_left(El_op,lam[j],Tn[j])
        El_identity = mult_left(El_identity,lam[j],Tn[j])
    
    for r in range(1,max_distance+1):
        El_op2 = mult_left_op(El_op,lam[step*r],Tn[step*r],op2)
        El_identity = mult_left(El_identity,lam[step*r],Tn[step*r])
        
        Correlation.append(np.real(np.tensordot(El_op2,Env_right[step*r],([0,1],[0,1]))/np.tensordot(El_identity,Env_right[step*r],([0,1],[0,1]))))
        if r < max_distance:
            El_op = mult_left(El_op,lam[step*r],Tn[step*r])
            for j in range(1,step):
                El_op = mult_left(El_op,lam[step*r + j],Tn[step*r + j])
                El_identity = mult_left(El_identity,lam[step*r + j],Tn[step*r + j])
    return Correlation

def Calc_mag(Env_left,Env_right,Tn,lam):

    N = len(Tn)
    m = Tn[0].shape[0]
    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i    
    
    mz = np.zeros(N)
    for i in range(N):
        mz[i]=np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz)/Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i]))
        
    return mz

def Calc_dot(Env_left,Env_right,Tn,lam,Sz,Sp,Sm):
    N = len(Tn)
    zz = np.zeros(N-1)
    pm = np.zeros(N-1)
    mp = np.zeros(N-1)
    for i in range(N-1):
        norm = Contract_two_site_no_op(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1])
        zz[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sz,Sz)/norm)
        pm[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sp,Sm)/norm)
        mp[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sm,Sp)/norm)

    return zz,pm,mp

def Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D):
    N = len(Tn)
    m = Tn[0].shape[0]
    
    Sp = np.zeros((m,m))
    for i in range(1,m):
        Sp[i-1,i] = np.sqrt(i * (m - i))
    
    Sm = np.zeros((m,m))
    for i in range(0,m-1):
        Sm[i+1,i] = np.sqrt((i + 1.0) * (m - 1.0 - i))

    Sz = np.zeros((m,m))
    for i in range(m):
        Sz[i,i] = 0.5 * (m - 1.0) - i
    
    Sx = 0.5 * (Sp + Sm)
    Sz2 = np.dot(Sz,Sz)

    Id = np.identity(m)

    zz = np.zeros(N-1)
    pm = np.zeros(N-1)
    mp = np.zeros(N-1)

    mx = np.zeros(N)
    z2 = np.zeros(N)
    
    for i in range(N):
        norm = Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i])
        mx[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sx)/norm)
        z2[i] = np.real(Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz2)/norm)
    
    for i in range(N-1):
        norm = Contract_two_site_no_op(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1])
        zz[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sz,Sz)/norm)
        pm[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sp,Sm)/norm)
        mp[i] = np.real(Contract_two_site(Env_left[i],Env_right[i+1],Tn[i],Tn[i+1],lam[i],lam[i+1],Sm,Sp)/norm)
    
    E = (Jz * np.sum(zz) + 0.5 * Jxy * (np.sum(pm) + np.sum(mp)) -hx * np.sum(mx) + D * np.sum(z2)) / (N-1)
    return E

def TEBD_Simulation(m,Jz,Jxy,hx,D,N,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):
    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step/output_dyn_num

    
    Ham = set_Hamiltonian_S(m,Jz,Jxy,hx,D)
    Ham_l = set_Hamiltonian_S(m,Jz,Jxy,hx,D,position=-1)
    Ham_r = set_Hamiltonian_S(m,Jz,Jxy,hx,D,position=1)
    
    ## Trivial Canonical form initial condition
    Tn = []
    lam = []
    for i in range(N):
        Tn.append(np.zeros((m,1,1),dtype=tensor_dtype))
        if ( i % 2 == 0):
            Tn[i][0,0,0] = 1.0
        else:
            Tn[i][m-1,0,0] = 1.0
        lam.append(np.ones((1),dtype=float))

    lam.append(np.ones((1),dtype=float))
        
    ## Imaginary time evolution (TEBD algorithm)
    tau = tau_max
    T = 0.0
    T_list = []
    E_list = []
    mz_list = []

    
    for n in range(tau_step):
        ## Imaginary time evolution operator U
        expH_l = linalg.expm(-tau*Ham_l).reshape(m,m,m,m)
        expH_c = linalg.expm(-tau*Ham).reshape(m,m,m,m)
        expH_r = linalg.expm(-tau*Ham_r).reshape(m,m,m,m)

        expH = [expH_l]
        for i in range(1,N-2):
            expH.append(expH_c)
        expH.append(expH_r)

        if (second_ST):
            expH2_l = linalg.expm(-0.5 * tau*Ham_l).reshape(m,m,m,m)
            expH2_c = linalg.expm(-0.5 * tau*Ham).reshape(m,m,m,m)
            expH2_r = linalg.expm(-0.5 * tau*Ham_r).reshape(m,m,m,m)

            expH2 = [expH2_l]
            for i in range(1,N-2):
                expH2.append(expH2_c)
            expH2.append(expH2_r)
        
        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment(Tn,lam)
            
            mz = Calc_mag(Env_left,Env_right,Tn,lam)
            E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz, Jxy,hx,D)
            print("##Dyn " + repr(T) + " "+ repr(E) + " "+repr(np.sqrt(np.sum(mz**2)/N))+" " +repr(mz))
            
            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            
        if second_ST:
            Tn,lam = TEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=inv_precision)
        else:
            Tn,lam = TEBD_update(Tn,lam,expH,chi_max,inv_precision=inv_precision)

        T += tau 
        tau = tau*tau_factor

    if output:
        Env_left,Env_right = Calc_Environment(Tn,lam)        
        mz = Calc_mag(Env_left,Env_right,Tn,lam)    
        E = Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)
        print(repr(m)+" " +repr(Jz) + " " + repr(Jxy)+ " "+repr(hx) + " " + repr(D) + " " +repr(E) + " " + repr(np.sqrt(np.sum(mz**2)/N)))

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='TEBD siumulator for one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N (default = 10)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
                        help='amplitude for SzSz interaction  (default = 1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
                        help='amplitude for SxSx + SySy interaction  (default = 1.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=3,
                        help='Spin size m=2S +1  (default = 3)' )
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
                        help='extarnal magnetix field  (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation  (default = 20)')
    parser.add_argument('-tau_max', metavar='tau_max',dest='tau_max', type=float, default=0.1,
                        help='start imaginary time step  (default = 0.1)')
    parser.add_argument('-tau_min', metavar='tau_min',dest='tau_min', type=float, default=0.001,
                        help='final imaginary time step  (default = 0.001)')
    parser.add_argument('-tau_step', metavar='tau_step',dest='tau_step', type=int, default=2000,
                        help='ITE steps. tau decreses from tau_max to tau_min gradually  (default = 2000)')
    parser.add_argument('-inv_precision', metavar='inv_precision',dest='inv_precision', type=float, default=1e-10,
                        help='smaller singular values than inv_precision is neglected at iTEBD update  (default = 1e-10)')
    parser.add_argument('--use_complex', action='store_const', const=True,
                        default=False, help='Use complex tensors  (default = False)')
    parser.add_argument('--second_ST', action='store_const', const=True,
                        default=False, help='Use second order Suzuki Trotter decomposition  (default = False)')
    parser.add_argument('--output_dyn', action='store_const', const=True,
                        default=False, help='Output energies along ITE  (default = False)')
    parser.add_argument('-output_dyn_num', metavar='output_dyn_num',dest='output_dyn_num', type=int, default=100,
                        help='number of data points at dynamics output  (default = 100)')
    return parser.parse_args()
    

if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    if args.use_complex:
        tensor_dtype = np.dtype(complex)
    else:
        tensor_dtype = np.dtype(float)

    if args.output_dyn:
        Tn, lam, T_list, E_list, mz_list = TEBD_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.D,args.N,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
    else:
        Tn, lam = TEBD_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.D,args.N,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)

