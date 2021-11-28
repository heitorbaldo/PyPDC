# -*- coding:utf-8 -*-


from numpy import array,zeros,eye,kron,dot,exp,pi,cos,sin,diag,ones,tril,resize,finfo
from numpy.core.fromnumeric import reshape
from numpy.core.umath import isnan
from numpy.lib.scimath import log, sqrt
from numpy.linalg import pinv, inv, eig
from scipy.linalg.basic import det
from scipy.linalg import schur
# from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from scipy.linalg import solve_sylvester

# scipy.linalg.solve_sylvester(a, b, q)[source]
# Computes a solution (X) to the Sylvester equation (AX + XB = Q).

eps = finfo(float).eps.item()

def nstrand(u, maxp = 30, simplep = True):
    '''
    %   Calculate the coeficients of multi-channel auto-regressive matrix using
    %   Nuttall-Strand algorithm (a generalization of single channel harmonic
    %                             method)
    %
    %   Input parameters:
    %     IP     - Ordem of autoregressive model (integer)
    %     u      - Complex matrix with NUMCHS channels of sample data
    %
    %   Output parameters:
    %     PF     - Covariance matrix of NUMCHS x NUMCHS of linear forward
    %              prediction error
    %     A      - Complex array of forward linear prediction matrix
    %              coefficients
    %     PB     - Complex backward linear prediction error covariance array
    %     B      - Complex array of backward linear prediction matrix
    %              coefficients
    '''
    [lx,cx]=u.shape
    if lx > cx:
        lx, cx = cx, lx  # Swapping the number of channels and record length
        u = u.T
        print ('Input matrix is probably transposed.')
        return
    NUMCHS=lx      # Number of channels.
    MAXORDER=200   # Maximum order of AR model allowed for calculation.
    N=max(u.shape) # N - Number of samples per channel.
    IP = maxp

    #    Initialization
    ISTAT=0
    if (IP > MAXORDER):
        ISTAT=3
        print('IP > 200')
        return

    ef=array(u)                    # Eq. (15.91)
    eb=array(u)                    # Eq. (15.91)
    pf=dot(u, u.transpose())       # Eq. (15.90)
    pb=array(pf)                   # Eq. (15.90)
    M=0
    #    Main Loop
    while 1:
        # Update estimated covariance errors  Eq. (15.89)
        pfhat=dot(ef[:,M+1:N],ef[:,M+1:N].transpose())
        pbhat=dot(eb[:,M:N-1],eb[:,M:N-1].transpose())
        pfbhat=dot(ef[:,M+1:N],eb[:,M:N-1].transpose())
        M=M+1
        # Calculate estimated partial correlation matrix - Eq. (15.98)
        #              (Nuttall-Strand algorithm only)
        RHO=solve_sylvester(dot(pfhat,inv(pf)),dot(inv(pb),pbhat),2*pfbhat)
        #RHO=solve_lyapunov(dot(pfhat,inv(pf)),dot(inv(pb),pbhat),-2*pfbhat)
        #  Update forward and backward reflection coeficients
        #% Eqs. (15.73),(15.74),(15.78) (Nuttall-Strand algorithm)
        AM=dot(-RHO,inv(pb))
        BM=dot(-RHO.transpose(),inv(pf))
        dimA=AM.shape[0]
        dimB=BM.shape[0]
        if M == 1:
            A=zeros((1,dimA,dimA),float)
            B=zeros((1,dimB,dimB),float)
        else:
            A=resize(A,(M,dimA,dimA))
            B=resize(B,(M,dimB,dimB))
        A[M-1,:,:]=AM
        B[M-1,:,:]=BM
        #  Update forward and backward covariance error  - Eqs. (15.75),(15.76)
        pf=pf-dot(dot(AM,BM),pf)
        pb=pb-dot(dot(BM,AM),pb)

        #  Update forward and backward predictor coeficients - Eqs.(15.84),(15.85)
        if not (M == 1):
            for K in range(1,M):
                temp1=A[K-1,:,:].copy()
                A[K-1,:,:]=A[K-1,:,:]+dot(AM,B[M-K-1,:,:])
                B[M-K-1,:,:]=B[M-K-1,:,:]+dot(BM,temp1)

        Tef=array(ef)
        ef[0:NUMCHS,range(N-1,M-1,-1)]=ef[:,range(N-1,M-1,-1)]+dot(AM,eb[:,range(N-2,M-2,-1)])
        eb[0:NUMCHS,range(N-1,M-1,-1)]=eb[:,range(N-2,M-2,-1)]+dot(BM,Tef[:,range(N-1,M-1,-1)])

        #  Verify if model order is adequate
        if M == IP:
            A=-A
            B=-B
            break

    if (simplep):
        return A.transpose(1,2,0), abs(pf)/N
    else:
        return pf,A,pb,B,ef,eb,ISTAT

def mvar(u, MaxIP = 30, alg=1, criterion=1, return_ef = False):
    '''
    %
    %[IP,pf,A,pb,B,ef,eb,vaic,Vaicv] = mvar(u,MaxIP,alg,criterion,return_ef)
    %
    % input: u     - data rows
    %        MaxIP - externally defined maximum IP (default = 30)
    %        alg   - for algorithm (1: Nutall-Strand; 2: mlsm; 3: Vieira-Morf; 
    %                               4: QR artfit)
    %        criteria for order choice - 0: MDL (not implemented);
    %                                     1: AIC; 2: Hanna-Quinn; 3: BIC-Schwarz;
    %                                     4: FPE; 5: fixed order in MaxIP
    %                                     6(not yet): estimate up to max order
    %                                     Negative(not yet) - keep criterion changes
    % output: 
    %
    %% Description
    %
    %
    %
    %
    %% References
    %
    % [1] Lutkepohl H (2005). New Introduction to Multiple Time Series Analysis. 
    %                         Springer-Verlag. 
    '''

    # StopFlag = False
    [nSegLength,nChannels] = u.transpose().shape

    # if criterion < 0:
    #     stopFlag = True
    #     criterion=abs(criterion)

    if criterion == 5:    # Fixed order at IP
        IP = maxIP
        if alg == 1:
            # print('u: ',u, 'MaxIP: ', MaxIP)
            npf, na, npb, nb, nef, neb, ISTAT = nstrand(u,MaxIP,False)
            pf =  npf/nSegLength
            A = na.transpose(1,2,0)
        else:
            print('Algorithm not implemented yet in Python.')

        vaic = max(u.shape)*log(det(pf)) + 2*nChannels*nChannels*IP # Akaike's 
        Vaicv =  vaic
        # if (not return_ef):
        return IP, pf, A, npb, nb, nef,neb,vaic,Vaicv
        # else:
        #     return na.transpose(1,2,0), nef

    vaicv=0
    if MaxIP == 0:
        MaxOrder = 30
        UpperboundOrder = round(3*sqrt(nSegLength)/nChannels)
        #% Marple Jr. page 409
        #% Suggested by Nuttall, 1976.
        UpperboundOrder = min([MaxOrder, UpperboundOrder])
    else:
        MaxOrder=MaxIP
        UpperboundOrder=MaxIP

    IP=1
    Vaicv=zeros((MaxOrder+1,1), float)

    while IP <= UpperboundOrder:
        if alg == 1:
            npf, na, npb, nb, nef, neb, ISTAT = nstrand(u,IP,False)
        else:
            print('ALG TO BE IMPLEMENTED: \n 2: Least squares estimator; \n 3: Vieira-Morf algorithm; \n 4: ARfit estimator.')

        if criterion == 1:   # Akaike's Information Criterion (AIC)  # (4.3.2)(Lutkepöhl '85)
            vaic = max(u.shape)*log(det(npf)) + 2*nChannels*nChannels*IP 

        elif criterion == 2: # Hanna-Quin (HQ)                       # (4.3.8)(Lutkepöhl '85)
            vaic = max(u.shape) * log(det(npf)) + 2*log(log(length(max(u.shape))))*nChannels*nChannels*IP

        elif criterion == 3: # Schwartz (1978) (SC or BIC criterion) # (4.3.9)(Lutkepöhl '85)
            vaic = max(u.shape) * log(det(npf)) + log(max(u.shape))*nChannels*nChannels*IP

        elif criterion == 4: # Final prediction error (Akaike, 1970) # (4.3.1)(Lutkepöhl '85)
            vaic = log(det(npf) * ((max(u.shape) + nChannels*IP+1)/(max(u.shape)-nChannels*IP-1))**nChannels)
        else:
            print('CRITERION TO BE IMPLEMENTED: ')

        Vaicv[IP,0] = vaic
        print('IP=', IP, '; vaic=', vaic)

        if (vaic > vaicv) and not (IP == 1): # Akaike condition
            vaic = vaicv
            print('vaic > vaicv ==> break')
            break
            # if not StopFlag:
            #     break

        vaicv = vaic
        pf = array(npf)
        A  = array(na)
        ef = array(nef)
        if alg == 1 or alg == 3:  # alg = Nutall-Strand and Vieira-Morf
            B  = array(nb)
            eb = array(neb)
            pb = array(npb)
        else:       # review status for backward prediction in clmsm
            B = []  # empty
            eb = [] # 
            pb = [] #

        IP=IP+1

    IP = IP - 1
    vaic = vaicv
    Vaicv = Vaicv[range(1,IP+1),0]
    Vaicv.shape = (Vaicv.size,1)

    if alg == 1:
        pf =  pf/nSegLength
        A = A.transpose(1,2,0)
    else:
        print('Algorithm not implemented yet in Python.')

    if (not return_ef):
        return IP, pf, A, npb, nb, nef, neb, vaic, Vaicv
    else:
        return IP, pf, A, npb, nb, nef, neb, vaic, Vaicv