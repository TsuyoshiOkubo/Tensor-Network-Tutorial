## Bug fixed on Dec. 2021

import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import TEBD
import argparse


def mult_left_Inv(v,lam_i,lam_i_inv,Tn_i,Tn_i_inv):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i_inv),(0,0)),np.diag(lam_i),(0,0)),Tn_i_inv,(0,2)),Tn_i.conj(),([0,1],[1,0]))

def mult_left_op(v,lam_i,Tn_i,op):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i,(0,1)),op,(1,1)),Tn_i.conj(),([0,2],[1,0]))

def mult_left_op_TR(v,lam_i,Tn_i,op):
    ## v has a shape v(chi,chi)
    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,np.diag(lam_i),(0,0)),np.diag(lam_i),(0,0)),Tn_i.conj(),(0,1)),op,(1,1)),Tn_i.conj(),([0,2],[1,0]))

def iTEBD_update(Tn,lam,expH,chi_max,inv_precision=1e-10):
    period = len(Tn)
    m = expH.shape[0]

    lam_inv =[]

    for i in range(period):
        lam_inv.append(1.0/lam[i])

    for eo in range(2):
        for i in range(eo,period,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[(i+1)%period].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[(i+1)%period]),(2,0))
                        ,Tn[(i+1)%period],(2,1))
                    ,np.diag(lam[(i+2)%period]),(3,0))
                ,expH,([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(chi_l*m,chi_r*m)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[(i+1)%period] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[(i+1)%period] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[(i+2)%period]),(1,0)).transpose(1,0,2)

            lam_inv[(i+1)%period] = 1.0/lam[(i+1)%period]

    return Tn,lam

def iTEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=1e-10):
    period = len(Tn)
    m = expH.shape[0]

    lam_inv =[]

    expH_eo=[expH2,expH]
    
    for i in range(period):
        lam_inv.append(1.0/lam[i])

    for eoe in range(3):
        if eoe == 1:
            eo = 1
        else:
            eo = 0    
        for i in range(eo,period,2):
            ## apply expH
            chi_l = Tn[i].shape[1]
            chi_r = Tn[(i+1)%period].shape[2]

            Theta = np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                np.diag(lam[i]),Tn[i],(1,1))
                            ,np.diag(lam[(i+1)%period]),(2,0))
                        ,Tn[(i+1)%period],(2,1))
                    ,np.diag(lam[(i+2)%period]),(3,0))
                ,expH_eo[eo],([1,2],[2,3])
            ).transpose(0,2,1,3).reshape(m*chi_l,m*chi_r)
            ## SVD
            U,s,VT = linalg.svd(Theta,full_matrices=False)

            ## Truncation
            ## smaller singular values are neglected. (So, actual chi varies depending on its bond)
            chi = np.min([np.sum(s > inv_precision),chi_max])
            lam[(i+1)%period] = s[:chi]/np.sqrt(np.sum(s[:chi]**2))

            Tn[i] = np.tensordot(np.diag(lam_inv[i]),U[:,:chi].reshape(chi_l,m,chi),(1,0)).transpose(1,0,2)
            Tn[(i+1)%period] = np.tensordot(VT[:chi,:].reshape(chi,chi_r,m),np.diag(lam_inv[(i+2)%period]),(1,0)).transpose(1,0,2)

            lam_inv[(i+1)%period] = 1.0/lam[(i+1)%period]

    return Tn, lam

def Calc_Environment_infinite(Tn,lam,canonical=False):
    ## Calculate left and right dominant eigen vectors

    period = len(Tn)
            
    Env_left = []
    Env_right = []
    if canonical:
        ## assume MPS is in canonical form
        for i in range(period):
            Env_left.append(np.identity((lam[i].shape[0])))
            Env_right.append(np.dot(np.dot(np.diag(lam[(i+1)%period]),np.identity((lam[(i+1)%period].shape[0]))),np.diag(lam[(i+1)%period])))
    else:
        for i in range(period):
            chi_l = Tn[i].shape[1]
            chi_r = Tn[i].shape[2]
            def vAB(v):
                v_new = TEBD.mult_left(v.reshape(chi_l,chi_l),lam[i],Tn[i])
                for j in range(1,period):
                    v_new = TEBD.mult_left(v_new,lam[(i+j)%period],Tn[(i+j)%period])

                return v_new.reshape(chi_l**2)
            def BAv(v):
                v_new = TEBD.mult_right(v.reshape(chi_r,chi_r),lam[(i+1)%period],Tn[i])
                for j in range(1,period):
                    v_new = TEBD.mult_right(v_new,lam[(i+period-j+1)%period],Tn[(i+period-j)%period])
                return v_new.reshape(chi_r**2)


            if (chi_r > 1):
                T_mat = spr_linalg.LinearOperator((chi_r**2,chi_r**2),matvec=BAv)

                eig_val,eig_vec = spr_linalg.eigs(T_mat,k=1)
            else:
                eig_vec = np.ones(1)

            Env_right.append(np.dot(np.dot(np.diag(lam[(i+1)%period]),eig_vec.reshape(chi_r,chi_r)),np.diag(lam[(i+1)%period])))

            if (chi_l > 1):
                T_mat = spr_linalg.LinearOperator((chi_l**2,chi_l**2),matvec=vAB)

                eig_val,eig_vec = spr_linalg.eigs(T_mat,k=1)
            else:
                eig_vec = np.ones(1)    

            Env_left.append(eig_vec.reshape(chi_l,chi_l))

    return Env_left,Env_right

def Eingenval_of_Transfer_Matrix(Tn,lam,e_num=5):
    ## Output e_num leading eigenvalus of the Transfer Matrix
    period = len(Tn)
    chi_r = Tn[0].shape[2]
    if chi_r > 2:
        def BAv(v):
            v_new = TEBD.mult_right(v.reshape(chi_r,chi_r),lam[1],Tn[0])
            for j in range(1,period):
                v_new = TEBD.mult_right(v_new,lam[(period-j+1)%period],Tn[(period-j)%period])
            return v_new.reshape(chi_r**2)
        T_mat = spr_linalg.LinearOperator((chi_r**2,chi_r**2),matvec=BAv)

        return spr_linalg.eigs(T_mat,k=min(e_num,chi_r-1),return_eigenvectors=False)
    else:
        T_half = np.tensordot(Tn[0],np.diag(lam[1]),(2,0))
        T_mat = np.tensordot(T_half,T_half.conj(),(0,0)).transpose(0,2,1,3)
        for j in range(1,period):
            T_half = np.tensordot(Tn[j],np.diag(lam[(j+1)%period]),(2,0))
            T_mat = np.tensordot(np.tensordot(T_half,T_half.conj(),(0,0)),T_mat,([1,3],[0,1]))

        eig = linalg.eigvals(np.reshape(T_mat,(chi_r**2,chi_r**2)))
        if chi_r ==1:
            return eig
        else:
            if np.abs(eig[0]) > np.abs(eig[1]):
                return np.array([eig[1],eig[0]])
            else:
                return np.array([eig[0],eig[1]])

def Transfer_Matrix_Inv_site(Tn,lam):
    ## Output dominant eigen_vecotr of the Transfer Matrix 
    period = len(Tn)
    chi_l = Tn[0].shape[1]
    def vAB(v):
        v_new = mult_left_Inv(v.reshape(chi_l,chi_l),lam[0],lam[1],Tn[0],Tn[0])
        for j in range(1,period):
            v_new = mult_left_Inv(v_new,lam[j],lam[period-j],Tn[j],Tn[period-j])
            return v_new.reshape(chi_l**2)
    T_mat = spr_linalg.LinearOperator((chi_l**2,chi_l**2),matvec=vAB)

    return spr_linalg.eigs(T_mat,k=1)

def Transfer_Matrix_Inv_bond(Tn,lam):
    ## Output dominant eigen_vecotr of the Transfer Matrix 
    period = len(Tn)
    chi_l = Tn[0].shape[1]
    def vAB(v):
        v_new = mult_left_Inv(v.reshape(chi_l,chi_l),lam[0],lam[0],Tn[0],Tn[period-1])
        for j in range(1,period):
            v_new = mult_left_Inv(v_new,lam[j],lam[period-j],Tn[j],Tn[period-j-1])
            return v_new.reshape(chi_l**2)
    T_mat = spr_linalg.LinearOperator((chi_l**2,chi_l**2),matvec=vAB)

    return spr_linalg.eigs(T_mat,k=1)

def Transfer_Matrix_SPT(Tn,lam,op):
    ## Output dominant eigen_vecotr the Transfer Matrix with op
    period = len(Tn)
    chi_l = Tn[0].shape[1]
    def vAB(v):
        v_new = TEBD.mult_left_op(v.reshape(chi_l,chi_l),lam[0],Tn[0],op)
        for j in range(1,period):
            v_new = TEBD.mult_left_op(v_new,lam[j],Tn[j],op)
            return v_new.reshape(chi_l**2)
    T_mat = spr_linalg.LinearOperator((chi_l**2,chi_l**2),matvec=vAB)

    return spr_linalg.eigs(T_mat,k=1)

def Transfer_Matrix_SPT_TR(Tn,lam,op):
    ## Output dominant eigen_vecotr the Transfer Matrix with TR
    period = len(Tn)
    chi_l = Tn[0].shape[1]
    def vAB(v):
        v_new = mult_left_op_TR(v.reshape(chi_l,chi_l),lam[0],Tn[0],op)
        for j in range(1,period):
            v_new = mult_left_op_TR(v_new,lam[j],Tn[j],op)
            return v_new.reshape(chi_l**2)
    T_mat = spr_linalg.LinearOperator((chi_l**2,chi_l**2),matvec=vAB)

    return spr_linalg.eigs(T_mat,k=1)

def Contract_correlation_infinite(Env_left,Env_right,Tn,lam,op1,op2,max_distance,step=1):
    ## Output sequence of correlation <op1(0) op2(r)> for r <= max_distance.
    ## The unit of r is the period of unit cell
    ## r is increased by step

    period = len(Tn)
    Correlation=[]
    El = Env_left[0]
    Er = Env_right[0]
    
    El_op = TEBD.mult_left_op(El,lam[0],Tn[0],op1)
    El_identity = TEBD.mult_left(El,lam[0],Tn[0])
    for j in range(1,period):
        El_op = TEBD.mult_left(El_op,lam[j],Tn[j])
        El_identity = TEBD.mult_left(El_identity,lam[j],Tn[j])
    
    for r in range(1,max_distance+1):
        El_op2 = TEBD.mult_left_op(El_op,lam[0],Tn[0],op2)
        El_identity = TEBD.mult_left(El_identity,lam[0],Tn[0])
        
        Correlation.append(np.real(np.tensordot(El_op2,Er,([0,1],[0,1]))/np.tensordot(El_identity,Er,([0,1],[0,1]))))
        El_op = TEBD.mult_left(El_op,lam[0],Tn[0])
        for j in range(1,period):
            El_op = TEBD.mult_left(El_op,lam[j],Tn[j])
            El_identity = TEBD.mult_left(El_identity,lam[j],Tn[j])
        
    return Correlation

def Calc_mag_infinite(Env_left,Env_right,Tn,lam):
    return TEBD.Calc_mag(Env_left,Env_right,Tn,lam)

def Calc_dot_infinite(Env_left,Env_right,Tn,lam,Sz,Sp,Sm):
    period = len(Tn)
    zz = np.zeros(period)
    pm = np.zeros(period)
    mp = np.zeros(period)
    for i in range(period):
        norm = TEBD.Contract_two_site_no_op(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period])
        zz[i] = np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sz,Sz)/norm)
        pm[i] = np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sp,Sm)/norm)
        mp[i] =np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sm,Sp)/norm)

    return zz,pm,mp

def Calc_Energy_infinite(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D):
    period = len(Tn)
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

    zz = np.zeros(period)
    pm = np.zeros(period)
    mp = np.zeros(period)

    mx = np.zeros(period)
    z2 = np.zeros(period)

    for i in range(period):
        norm = TEBD.Contract_one_site_no_op(Env_left[i],Env_right[i],Tn[i],lam[i])
        mx[i] = np.real(TEBD.Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sx)/norm)
        z2[i] = np.real(TEBD.Contract_one_site(Env_left[i],Env_right[i],Tn[i],lam[i],Sz2)/norm)

    for i in range(period):
        norm = TEBD.Contract_two_site_no_op(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period])
        zz[i] = np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sz,Sz)/norm)
        pm[i] = np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sp,Sm)/norm)
        mp[i] =np.real(TEBD.Contract_two_site(Env_left[i],Env_right[(i+1)%period],Tn[i],Tn[(i+1)%period],lam[i],lam[(i+1)%period],Sm,Sp)/norm)

    E = (Jz * np.sum(zz) + 0.5 * Jxy * (np.sum(pm) + np.sum(mp)) -hx * np.sum(mx) + D * np.sum(z2)) / period
    return E

def iTEBD_Simulation(m,Jz,Jxy,hx,D,chi_max,tau_max,tau_min,tau_step,inv_precision=1e-10,second_ST=False,tensor_dtype=np.dtype(float),output_dyn=False,output_dyn_num=100,output=False):
    tau_factor = (tau_min/tau_max)**(1.0/tau_step)
    output_step = tau_step/output_dyn_num
    
    Ham = TEBD.set_Hamiltonian_S(m,Jz,Jxy,hx,D,position=0)
    period = 2 ## number of independent matrices (2 is the standard iTEBD)

    ## Trivial Canonical form initial condition
    Tn = []
    lam = []
    for i in range(period):
        Tn.append(np.zeros((m,1,1),dtype=tensor_dtype))
        if ( i % 2 == 0):
            Tn[i][0,0,0] = 1.0
        else:
            Tn[i][m-1,0,0] = 1.0
        lam.append(np.ones((1),dtype=float))
    
    ## Imaginary time evolution (iTEBD algorithm)
    tau = tau_max
    T = 0.0
    if output_dyn:
        T_list = []
        E_list = []
        mz_list = []
    
    for n in range(tau_step):
        ## Imaginary time evolution operator U
        expH = linalg.expm(-tau*Ham).reshape(m,m,m,m)
        if (second_ST):
            expH2 = linalg.expm(-0.5*tau*Ham).reshape(m,m,m,m)

        if (output_dyn and n%output_step == 0):
            Env_left,Env_right = Calc_Environment_infinite(Tn,lam)
            
            mz = Calc_mag_infinite(Env_left,Env_right,Tn,lam)
            E = Calc_Energy_infinite(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)
            print("##Dyn "+repr(T) + " " +repr(E) + " " + repr(np.sqrt(np.sum(mz**2)/period)) + " " + repr(mz))
            T_list.append(T)
            E_list.append(E)
            mz_list.append(mz)
            

        if second_ST:
            Tn,lam = iTEBD_update_second(Tn,lam,expH,expH2,chi_max,inv_precision=inv_precision)
        else:
            Tn,lam = iTEBD_update(Tn,lam,expH,chi_max,inv_precision=inv_precision)

        T += tau 
        tau = tau*tau_factor


    if output:
        Env_left,Env_right = Calc_Environment_infinite(Tn,lam)
        mz = Calc_mag_infinite(Env_left,Env_right,Tn,lam)
        E = Calc_Energy_infinite(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)

        print(repr(m)+" " + repr(Jz) + " "+ repr(Jxy)+ " "+repr(hx)+ " "+repr(D)+ " "+repr(E) + " "+repr(np.sqrt(np.sum(mz**2)/period)))

    if output_dyn:
        return Tn,lam,np.array(T_list),np.array(E_list),np.array(mz_list)
    else:
        return Tn,lam


#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='iTEBD siumulator for infinite one dimensional spin model')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
                        help='anisotoropy for SzSz interaction')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
                        help='anisotoropy for SxSx+SySy interaction')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=3,
                        help='Spin size m=2S +1')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
                        help='extarnal magnetix field')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation')
    parser.add_argument('-tau_max', metavar='tau_max',dest='tau_max', type=float, default=0.1,
                        help='start imaginary time step')
    parser.add_argument('-tau_min', metavar='tau_min',dest='tau_min', type=float, default=0.001,
                        help='final imaginary time step')
    parser.add_argument('-tau_step', metavar='tau_step',dest='tau_step', type=int, default=2000,
                        help='ITE steps. tau decreses from tau_max to tau_min gradually')
    parser.add_argument('-inv_precision', metavar='inv_precision',dest='inv_precision', type=float, default=1e-10,
                        help='smaller singular values than inv_precision is neglected at iTEBD update')
    parser.add_argument('--use_complex', action='store_const', const=True,
                        default=False, help='Use complex tensors')
    parser.add_argument('--second_ST', action='store_const', const=True,
                        default=False, help='Use second order Suzuki Trotter decomposition')
    parser.add_argument('--output_dyn', action='store_const', const=True,
                        default=False, help='Output energies along ITE')
    parser.add_argument('-output_dyn_num', metavar='output_dyn_num',dest='output_dyn_num', type=int, default=100,
                        help='number of data points at dynamics output')
    return parser.parse_args()

if __name__ == "__main__":
    ## read params from command line
    args = parse_args()

    if args.use_complex:
        tensor_dtype = np.dtype(complex)
    else:
        tensor_dtype = np.dtype(float)

    if args.output_dyn:
        Tn, lam,T_list,E_list,mz_list = iTEBD_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.D,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)
    else:
        Tn, lam = iTEBD_Simulation(args.m,args.Jz,args.Jxy,args.hx,args.D,args.chi_max,args.tau_max,args.tau_min,args.tau_step,args.inv_precision,args.second_ST,tensor_dtype,args.output_dyn,args.output_dyn_num,output=True)


