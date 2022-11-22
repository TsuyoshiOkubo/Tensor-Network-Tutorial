import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse


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
    m = expH.shape[0]

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
                ,expH,([1,2],[2,3])
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


def Calc_Environment(Tn,lam,canonical=False):
    ## Calculate left and right contraction exactly

    N = len(Tn)
    Env_left = []
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

        Env_right = []
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


def calc_innerproduct(Tn1,lam1,Tn2,lam2):


    chi1 = Tn1[0].shape[2]
    chi2 = Tn2[0].shape[2]

    vec = np.tensordot(Tn1[0],Tn2[0].conj(),axes=(0,0)).reshape(chi1,chi2)

    for i in range(1,len(Tn1)):
        vec = np.tensordot(np.tensordot(np.tensordot(np.tensordot(vec,np.diag(lam1[i]),(0,0)),np.diag(lam2[i]),(0,0)),Tn1[i],(0,1)),Tn2[i].conj(),([0,1],[1,0]))

    return vec.reshape(1)[0]

def remake_vec(Tn,lam):

    chi = Tn[0].shape[2]
    m = Tn[0].shape[0]
    vec = np.reshape(Tn[0],(m,chi))

    for i in range(1,len(Tn)):
        vec = np.tensordot(np.tensordot(vec,np.diag(lam[i]),(i,0)),Tn[i],(i,1))
    return vec.flatten()

