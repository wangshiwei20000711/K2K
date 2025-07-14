import copy
from scipy.sparse import coo_matrix
from scipy import linalg
import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as plt


epst=1
while 1+epst!=1:
    eps=epst
    epst/=2
###############################################################################################################
# *********子程序[0]文本转数组************** #
def Func_txt_to_list(input_path, mode=0):
    path = input_path
    fp = open(path, encoding='utf-8', errors='ignore')
    lines = fp.readlines()
    NumLine = int(len(lines))
    fp.close()
    NumColumn = 0
    for i, line in enumerate(lines):
        temp = line.split()
        if len(temp) > NumColumn:
            NumColumn = len(temp)
    x = np.zeros((NumLine,NumColumn))
    for i, line in enumerate(lines):
        temp = line.split()
        for j in range(len(temp)):
            if mode==0:
                x[i][j]=temp[j]
            elif mode==1:
                if j%2==1:
                    x[i][j] = temp[j]
    return x
# ********************************************* #

# ************************ 子程序[1.1] ************************* #
def  corrcubic(theta, d):
    #CORRCUBIC  Cubic correlation function,
    m,n = d.shape # number of differences and dimension of data
    if  len(theta) == 1:
        theta = np.tile(theta,(1,n))
    elif  len(theta) != n:
        raise Exception('Length of theta must be 1 or ',n)
    else:
        theta = theta[:].T
    temp = abs(d) * np.tile(theta,(m,1))
    td = np.minimum(temp, np.ones_like(temp))
    ss = 1 - td**2 * (3 - 2*td)
    r = ss.prod(axis=1)

    dr = np.zeros([m,n])
    for j in range(n):
        dd = 6*theta[j] * np.sign(d[:,j]) * td[:,j] * (td[:,j] - 1)
        dr[:,j] = ss[:,np.append(np.arange(j),np.arange(j+1,n))].prod(axis=1) * dd
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.2] ************************* #
def  correxp(theta, d):
    #CORREXP  Exponential correlation function
    m,n = d.shape  # number of differences and dimension of data
    lt = len(theta)
    if  lt == 1:
        theta = np.tile(theta,(1,n))
    elif  lt != n:
        raise Exception('Length of theta must be 1 or',n)
    else:
        theta = theta.T
    td = abs(d) * np.tile(-theta, (m, 1))
    r = np.exp(sum(td,2))
    dr = np.tile(-theta,(m,1)) * np.sign(d) * np.tile(r,(1,n))
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.3] ************************* #
def  correxpg(theta, d):
    #CORREXPG  General exponential correlation function
    m,n = d.shape  # number of differences and dimension of data
    lt = len(theta)
    if  n > 1 & lt == 2:
      theta = np.array([np.tile(theta[1],(1,n)), theta[2]])
    elif  lt != n+1:
        raise Exception('Length of theta must be 2 or',n+1)
    else:
        theta = theta.T

    pow = theta[-1]
    tt = np.tile(-theta[1:n], (m, 1))
    td = abs(d)**pow * tt
    r = np.exp(td.sum(axis=1))
    dr = pow  * tt * np.sign(d) * (abs(d) ** (pow-1)) * np.tile(r,(1,n))
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.4] ************************* #
def  corrgauss(theta, d):
    #CORRGAUSS  Gaussian correlation function,
    m, n = d.shape  # number of differences and dimension of data
    if  len(theta) == 1:
        theta = np.tile(theta,(1,n))
    elif  len(theta) != n:
        raise Exception('Length of theta must be 1 or',n)
    td = d**2 * np.tile(-theta.T,(m,1))
    r = np.exp(td.sum(axis=1))
    dr = np.tile(-2*theta.reshape(-1,1).T,(m,1)) * d * np.tile(r.reshape(-1,1),(1,n))
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.5] ************************* #
def corrlin(theta, d):
    #CORRLIN  Linear correlation function,
    m,n = d.shape  # number of differences and dimension of data
    if  len(theta) == 1:
        theta = np.tile(theta,(1,n))
    elif  len(theta) != n:
        raise Exception('Length of theta must be 1 or',n)

    td = max(1 - abs(d) * np.tile(theta.T,(m,1)), 0)
    r = td.prod(axis=1)
    dr = np.zeros([m,n])
    for j  in range(n):
        dr[:,j] = td[:,np.append(np.arange(j),np.arange(j+1,n))].prod(axis=1) * (-theta[j] * np.sign(d[:,j]))
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.6] ************************* #
def  corrspherical(theta, d):
    #CORRSPHERICAL  Spherical correlation function,
    m,n = d.shape  # number of differences and dimension of data
    if  len(theta) == 1:
        theta = np.tile(theta,(1,n))
    elif  len(theta) != n:
        raise Exception('Length of theta must be 1 or #d',n)
    else:
      theta = theta.T
    temp = abs(d) * np.tile(theta,(m,1))
    td = np.minimum(temp, np.ones_like(temp))
    ss = 1 - td * (1.5 - .5*td**2)
    r = ss.prod(axis=1)
    dr = np.zeros([m,n])
    for  j in range(n):
        dd = 1.5*theta[j] * np.sign(d[:,j])*(td[:,j]**2 - 1)
        dr[:,j] = ss[:,np.append(np.arange(j),np.arange(j+1,n))].prod(axis=1) * dd
    return r, dr
# *********************************************************** #

# ************************ 子程序[1.7] ************************* #
def corrspline(theta, d):
    #CORRSPLINE  Cubic spline correlation function,
    m,n = d.shape  # number of differences and dimension of data
    if  len(theta) == 1:
        theta = np.tile(theta,(1,n))
    elif  len(theta) != n:
        raise Exception('Length of theta must be 1 or #d',n)
    else:
      theta = theta.T

    mn = m*n
    ss = np.zeros([mn,1])
    xi = abs(d)*np.tile(theta,(m,1)).reshape(mn,1)
    # Contributions to first and second part of spline
    i1 = np.where(xi <= 0.2)[0]
    i2 = np.where(0.2 < xi and xi < 1)[0]
    if  len(i1)!=0:
      ss[i1.astype(int)] = 1 - xi[i1.astype(int)]**2 * (15  - 30*xi[i1.astype(int)])
    if  len(i2)!=0:
      ss[i2.astype(int)] = 1.25 * (1 - xi[i2.astype(int)])**3
    # Values of correlation
    ss = ss.reshape(m,n)
    r = ss.prod(axis=1)

    u = (np.sign(d) * np.tile(theta,(m,1))).reshape(mn,1)
    dr = np.zeros(mn,1)
    if  len(i1[0])!=0:
        dr[i1.astype(int)] = u[i1.astype(int)] * ( (90*xi(i1.astype(int)) - 30) * xi(i1.astype(int)) )
    if  len(i2[0])!=0:
        dr[i2.astype(int)] = -3.75 * u[i2.astype(int)] * (1 - xi[i2.astype(int)])**2
    ii = np.arange(m)
    for j in range(n):
        sj = ss[:,j]
        ss[:,j] = dr[ii]
        dr[ii] = ss.prod(axis=1)
        ss[:,j] = sj
        ii = ii + m
    dr = dr.reshape(m,n)
    return r, dr
# *********************************************************** #

# ************************ 子程序[2.1] *********************** #
def  regpoly0(S): #用的式2.21
    #REGPOLY0  Zero order polynomial regression function
    m, n=S.shape
    f = np.ones([m,1])
    df = np.zeros([n,1])
    return f, df
# *********************************************************** #

# ************************ 子程序[2.2] ************************* #
def  regpoly1(S): #用的式2.22
    #REGPOLY1  First order polynomial regression function

    m, n= S.shape
    f = np.append(np.ones([m,1]),  S)
    df = np.append(np.zeros([n,1]), np.eye(n))
    return f, df
# *********************************************************** #

# ************************ 子程序[2.3] ************************* #
def  regpoly2(S): #用的式2.23
    #REGPOLY2  Second order polynomial regression function

    m, n = S.shape
    nn = int((n+1)*(n+2)/2)  # Number of columns in f
    # Compute  f
    f = np.append(np.append(np.ones([m,1]), S, axis=1), np.zeros([m,nn-n-1]), axis=1)
    j = n+1
    q = n
    for  k in range(n):
        f[:,j+np.arange(0,q)] = np.tile(S[:,k].reshape(-1,1),(1,q)) * S[:,k:n]
        j = j+q
        q = q-1
    df = np.append(np.append(np.zeros([n,1]), np.eye(n), axis=1), np.zeros([n,nn-n-1]),axis=1)
    j = n+1
    q = n
    for k in range(n):
        df[k,j-1+np.arange(0,q)] = np.append(2*S[0,k], S[0,k+1:n])
        for i in range(n-k-1):
            df[k+i,j+i] = S[0,k]
        j = j+q
        q = q-1
    return f, df
# *********************************************************** #


# ************************ 子程序[3] ************************* #
def dacefit(S, Y, regr, corr, theta0, lob=np.nan, upb=np.nan):
    # DACEFIT Constrained non-linear least-squares fit of a given correlation
    # model to the provided data set and regression model

    ## >>>>>>>>>>>>>>>>   Auxiliary functions  ====================
    def objfunc(theta, par):
        # Initialize
        obj = np.inf
        fit = {'sigma2': np.NaN, 'beta': np.NaN, 'gamma': np.NaN, 'C': np.NaN, 'Ft': np.NaN, 'G': np.NaN}  # fit建立结构体
        m = par.get('F').shape[0]
        r,_ = par.get('corr')(theta, par.get('D'))
        idx = np.where(r > 0)[0]
        o = np.arange(m).reshape(1, -1).T
        mu = (10 + m) * eps
        _data = np.append(r[idx].reshape(-1, 1), np.ones([m, 1]) + mu,axis=0).reshape(-1)
        _rol = np.append(par.get('ij')[idx, 0].reshape(-1, 1), o, axis=0).astype(int).reshape(-1)
        _col = np.append(par.get('ij')[idx, 1].reshape(-1, 1), o, axis=0).astype(int).reshape(-1)
        R = coo_matrix((_data, (_rol,_col)))
        R = R.toarray()
        R = R+R.T
        for ti in range(R.shape[0]):
            R[ti,ti] = R[ti,ti]/2
        if not np.all(np.linalg.eigvals(R) > 0):
            print('not zheng ding')
            return obj, fit
        C = linalg.cholesky(R)
        C = C.T
        Ft = np.linalg.solve(C, par.get('F'))  # 式3.11
        Q, G = sl.qr(Ft)
        gm, gn = G.shape
        if gn < gm:
            G = G[0:gn, :]
            Q = Q[:, 0:gn]
        if 1 / np.linalg.cond(G, p=1) < 1e-10:
            # Check   F
            _, temps, _= np.linalg.svd(par.get('F'))
            if temps.max() > 1e15:  #if np.linalg.cond(par.get('F'), p=1) > 1e15:
                raise Exception('F is too ill conditioned\nPoor combination of regression model and design sites')
            else:  # Matrix  Ft  is too ill conditioned
                return obj, fit
        Yt = np.linalg.solve(C, par.get('y'))
        beta = np.linalg.solve(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)
        sigma2 = sum(rho ** 2) / m
        detR = np.prod((np.diag(C)) ** (2 / m))
        obj = sum(sigma2) * detR
        fit = {'sigma2': sigma2, 'beta': beta, 'gamma': np.dot(rho.T,np.linalg.inv(C)), 'C': C, 'Ft': Ft,
               'G': G.T}
        return obj, fit

    # --------------------------------------------------------
    def boxmin(t0, lo, up, par):
        # BOXMIN  Minimize with positive box constraints使用正的封闭约束来最小化
        # Initialize
        t, f, fit, itpar = start(t0, lo, up, par)
        if np.isinf(f):
            p = len(t)
            if p <= 2:
                kmax = 2
            else:
                kmax = min(p, 4)
            for k in range(kmax):
                th = t
                t, f, fit, itpar = explore(t, f, fit, itpar, par)
                t, f, fit, itpar = move(th, t, f, fit, itpar, par)
        perf = {'nv': itpar.get('nv'), 'perf': itpar.get('perf')[:, 0:itpar.get('nv')]}
        return t, f, fit, perf

    # --------------------------------------------------------

    def start(t0, lo, up, par):
        # Get starting point and iteration parameters
        # Initialize
        t = copy.deepcopy(t0)
        p = len(t)
        D = 2 ** (np.arange(p).T / (p + 2))
        ee = np.where(up == lo)[0]  # Equality constraints
        if len(ee)!=0:
            D[ee.astype(int)] = np.ones([len(ee), 1])
            t[ee.astype(int)] = up[ee.astype(int)]
        ng = np.where(np.logical_or(np.greater(lo,t) ,np.greater(t,up))==True)[0]  # Free starting values
        if len(ng)!=0:
            t[ng.astype(int)] = (lo[ng.astype(int)]**(1/8) * up[ng.astype(int)] ** (7/ 8)) # Starting point
        ne = np.where(D != 1)[0]
        # Check starting point and initialize performance info
        f, fit = objfunc(t, par)
        nv = 1
        itpar = {'D': D, 'ne': ne, 'lo': lo, 'up': up, 'perf': np.zeros([p + 2, 200 * p]), 'nv': 1}
        itpar.get('perf')[:, 0] = np.append(np.append(t, f), 1)
        if np.isinf(f):  # Bad parameter region
            return t, f, fit, itpar
        # Try to improve starting guess
        if len(ng)!=0:
            d0 = 16
            d1 = 2
            q = len(ng)
            th = t
            fh = f
            jdom = ng[1].astype(int)
            for k in range(q):
                j = ng[k].astype(int)
                fk = fh
                tk = th
                DD = np.ones([p, 1])
                DD[ng.astype(int)] = np.tile(1 / d1, (q, 1))
                DD[j] = 1 / d0
                alpha = min(np.log(lo[ng.astype(int)] / th[ng.astype(int)]) / np.log(DD[ng.astype(int)])) / 5
                v = DD ** alpha
                for rept in range(4):  # 重复4次
                    tt = tk * v
                    ff, fitt = objfunc(tt, par)
                    nv = nv + 1
                    itpar.get('perf')[:, nv] = np.append(np.append(tt, ff), 1)
                    if ff <= fk:
                        tk = tt
                        fk = ff
                        if ff <= f:
                            t = tt
                            f = ff
                            fit = fitt
                            jdom = j
                    else:
                        itpar.get('perf')[-1, nv] = -1
                        break
            # Update Delta
            if jdom > 1:
                D[[1, jdom]] = D[[jdom, 1]]
                itpar.D = D
        # free variables
        itpar.update({'nv': nv})
        return t, f, fit, itpar

    # --------------------------------------------------------

    def explore(t, f, fit, itpar, par):
        # Explore step
        nv = itpar.get('nv')
        ne = itpar.get('ne')
        for k in range(len(ne)):
            j = ne[k]
            tt = t
            DD = itpar.get('D')[j]
            if t[j] == itpar.get('up')[j]:
                atbd = 1
                tt[j] = t[j] / np.sqrt(DD)
            elif t[j] == itpar.get('lo')[j]:
                atbd = 1
                tt[j] = t[j] * np.sqrt(DD)
            else:
                atbd = 0
                tt[j] = min(itpar.get('up')[j], t[j] * DD)
            ff, fitt = objfunc(tt, par)
            nv = nv + 1
            itpar.get('perf')[:, nv] = np.append(np.append(tt, ff), 2)
            if ff < f:
                t = tt
                f = ff
                fit = fitt
            else:
                itpar.get('perf')[-1, nv] = -2
                if atbd != 0:  # try decrease
                    tt[j] = max(itpar.get('lo')[j], t[j] / DD)
                    ff, fitt = objfunc(tt, par)
                    nv = nv + 1
                    itpar.get('perf')[:, nv] = np.append(np.append(tt, ff), 2)
                    if ff < f:
                        t = tt
                        f = ff
                        fit = fitt
                    else:
                        itpar.get('perf')[-1, nv] = -2
        # k
        itpar.update({'nv': nv})
        return t, f, fit, itpar

    # --------------------------------------------------------

    def move(th, t, f, fit, itpar, par):
        # Pattern move
        nv = itpar.get('nv')
        ne = itpar.get('ne')
        p = len(t)
        v = t / th
        if all(v == 1):
            itpar.update({'D': np.append(np.arange(1, p), 1) ** 0.2})
            return t, f, fit, itpar
        # Proper move
        rept = 1
        while rept:
            boo = np.greater(itpar.get('lo'), t * v).astype(int)
            coooo = (1-boo)*(t * v)+boo*itpar.get('lo')
            boo = np.greater(coooo, itpar.get('up')).astype(int)
            tt = (itpar.get('up')*boo+(1-boo)*coooo)
            ff, fitt = objfunc(tt, par)
            nv = nv + 1
            itpar.get('perf')[:, nv] = np.append(np.append(tt, ff), 3)
            if ff < f:
                t = tt
                f = ff
                fit = fitt
                v = v ** 2
            else:
                itpar.get('perf')[-1, nv] = -3
                rept = 0
            if np.logical_or(tt == itpar.get('lo') , tt == itpar.get('up')).any:
                rept = 0

        itpar.update({'nv': nv})
        itpar.update({'D': np.append(np.arange(1, p), 1) ** 0.25})
        return t, f, fit, itpar

    ## Check design points
    m,n = S.shape  # number of design sites and their dimension
    sY = Y.shape
    if min(sY) == 1:
        Y = Y.reshape(max(sY))
        lY = max(sY)
        sY = Y.shape
    else:
        lY = sY[0]
    if m != lY:
        raise Exception('S and Y must have the same number of rows')

    ## Check correlation parameters
    lth = len(theta0)
    if lob is not np.nan and upb is not np.nan:
        if len(lob) != lth or len(upb) != lth:
            raise Exception('theta0, lob and upb must have the same length')
        if any(lob < 0) or any(upb < lob):
            raise Exception('The bounds must satisfy  0 < lob <= upb')
    else:  # given theta
        if any(theta0 <= 0):
            raise Exception('theta0 must be strictly positive')

    ## Normalize data
    mS = np.mean(S,axis=0)
    sS = np.std(S,axis=0)
    mY = np.mean(Y,axis=0)
    sY = np.std(Y,axis=0)
    if len(Y.shape)==1:
        Y=Y.reshape(-1,1)
    # Check for 'missing dimension'
    j = (np.where(sS == 0))[0]
    if len(j)!=0:
        sS[j.astype(int)] = 1
    j = np.where(sY == 0)[0]
    if len(j)!=0:
        sY[j.astype(int)] = 1
    S = (S - np.tile(mS, (m, 1))) / np.tile(sS, (m, 1))
    Y = (Y - np.tile(mY, (m, 1))) / np.tile(sY, (m, 1))

    ## Calculate distances D between points
    mzmax = int((m) * (m - 1) / 2)  # number of non-zero distances
    ij = np.zeros([mzmax, 2])  # initialize matrix with indices
    D = np.zeros([mzmax, n])  # initialize matrix with distances
    ll = np.array([-1])
    for k in range(m-1):
        ll = ll[-1]+1+np.arange(0,m-(k+1))
        ij[ll,:] = np.append(np.tile(k, (m - (k+1), 1)),np.arange(k+1,m).T.reshape(-1,1),axis=1) # indices for sparse matrix
        D[ll,:] = np.tile(S[k,:], (m - (k+1), 1)) - S[k + 1: m,:]  # differences between points
    if min(abs(D).sum(axis=1)) == 0:
        raise Exception('Multiple design sites are not allowed')

    ## Regression matrix
    F,_ = regr(S)
    mF, p = F.shape
    if mF != m:
        raise Exception('number of rows in  F  and  S  do not match')
    if p > mF:
        raise Exception('least squares problem is underdetermined')

    ## parameters for objective function
    par = {'corr':corr, 'regr':regr, 'y': Y, 'F': F, 'D': D, 'ij': ij, 'scS': sS}

    ## Determine theta
    if lob is not np.nan and upb is not np.nan:
        # Bound constrained non-linear optimization
        theta,f,fit,perf = boxmin(theta0, lob, upb, par)
        if np.isinf(f):
            raise Exception('Bad parameter region.  Try increasing  upb')
        else:
            # Given theta
            theta = copy.deepcopy(theta0)
            f ,fit = objfunc(theta, par)
            perf = {'perf': np.append(np.append(theta, f), 1), 'nv': 1}
        if np.isinf(f):
            raise Exception('Bad point.  Try increasing theta0')

    ## Return values
    dmodel = {'regr': regr, 'corr': corr, 'theta': theta.T, 'beta': fit.get('beta'), 'gamma': fit.get('gamma'), 'sigma2': sY**2. * fit.get('sigma2'),
    'S': S, 'Ssc': np.append(mS.reshape(1,-1),sS.reshape(1,-1),axis=0), 'Ysc': np.append(mY.reshape(1,-1),sY.reshape(1,-1),axis=0), 'C': fit.get('C'), 'Ft': fit.get('Ft'), 'G': fit.get('G')}  # 生成输出的结构体DACE模型，各元素含义见Word文档

    return dmodel,perf
# *********************************************************** #

def predictor(x, dmodel):
    # PREDICTOR  Predictor for y(x) using the given DACE model.
    # >>>>>>>>>>>>>>>>   Auxiliary function  ====================
    def colsum(x):
        # Columnwise sum of elements in  x
        if x.shape[0] == 1:
            s = x
        else:
            s = x.sum(axis=0)
        return s

    or1 = np.NaN
    or2 = np.NaN
    dmse = np.NaN  # Default return values
    if np.isnan(dmodel.get('beta')).any==True:
        y = np.NaN
        raise Exception('DMODEL has not been found')
    m,n = (dmodel.get('S')).shape  # number of design sites and number of dimensions
    sx = x.shape  # number of trial sites and their dimension
    if len(sx) == 1 and n > 1:  # Single trial point
        nx = max(sx)
        if nx == n:
            mx = 1
            x = x.T
    else:
        mx = sx[0]
        nx = sx[1]
    if nx != n:
        raise Exception('Dimension of trial sites should be', n)

    # Normalize trial sites
    x = (x - np.tile(dmodel.get('Ssc')[0,:],(mx, 1)))/ np.tile(dmodel.get('Ssc')[1,:], (mx, 1))
    q = (dmodel.get('Ysc')).shape[1]  # number of response functions
    y = np.zeros([mx, q])  # initialize result

    if mx == 1:  # one site only
        dx = np.tile(x, (m, 1)) - dmodel.get('S')  # distances to design sites
        f, df =(dmodel.get('regr'))(x)
        r,dr = dmodel.get('corr')(dmodel.get('theta'), dx)
        # Scaled Jacobian
        dy = (np.dot(df , dmodel.get('beta'))).T + np.dot(dmodel.get('gamma') , dr)
        # Unscaled Jacobian
        or1 = dy* np.tile(dmodel.get('Ysc')[1,:].T.reshape(-1,1), (1, nx)) / np.tile(dmodel.get('Ssc')[1,:].reshape(1,-1), (q, 1))
        if q == 1:
            # Gradient as a column vector
            or1 = or1.T
        rt = np.linalg.solve(dmodel.get('C') , r)
        u = np.dot((dmodel.get('Ft')).T ,rt.reshape(-1,1)) - f.T
        v = np.linalg.solve(dmodel.get('G'), u)
        or2 = np.tile(dmodel.get('sigma2'), (mx, 1)) * np.tile((1 + sum(v**2) - sum(rt**2)).T.reshape(-1,1),(1,q))

        # Scaled gradient as a row vector
        Gv = np.linalg.solve(dmodel.get('G').T , v)
        g = np.dot((np.dot(dmodel.get('Ft') , Gv) - rt.reshape(-1,1)).T , np.linalg.solve(dmodel.get('C') , dr)) - (np.dot(df , Gv)).T
        # Unscaled Jacobian
        dmse = np.tile(2 * dmodel.get('sigma2').T.reshape(-1,1),(1,nx)) * np.tile((g / dmodel.get('Ssc')[1,:]).reshape(1,-1),(q,1))
        if q == 1:
            # Gradient as a column vector
            dmse = dmse.T
        # Scaled predictor
        sy = np.dot(f , dmodel.get('beta')) + (np.dot(dmodel.get('gamma') , r)).T
        # Predictor
        y = (dmodel.get('Ysc')[0,:] + dmodel.get('Ysc')[1,:]*sy).T
        y = y.reshape(-1)
    else:  # several trial sites
        raise Exception('Only predict 1 datapoint at 1 time.')
        # Get distances to design sites
        dx = np.zeros([mx * m, n])
        kk = np.arange(m)
        for k in range(mx):
            dx[kk,:] = np.tile(x[k,:], (m, 1)) - dmodel.get('S')
            kk = kk + m
        # Get regression function and correlation
        f,_ = (dmodel.get('regr'))(x)
        r,_ = (dmodel.get('corr'))(dmodel.get('theta'), dx)
        r = r.reshape([m, mx])

        # Scaled predictor
        sy = np.dot(f , dmodel.get('beta')) + (np.dot(dmodel.get('gamma') , r)).T
        # Predictor
        y = np.tile(dmodel.get('Ysc')[0,:], (mx, 1)) + np.tile(dmodel.get('Ysc')[1,:], (mx, 1))*sy

        rt = np.linalg.solve(dmodel.get('C') , r)
        u = np.linalg.solve(dmodel.get('G') , (np.dot(dmodel.get('Ft').T , rt) - f.T))
        or1 = np.tile(dmodel.get('sigma2').reshape(1,-1), (mx, 1)) * np.tile((1 + colsum(u**2) - colsum(rt**2)).T.reshape(-1,1),(1,q))
        or2 = None
        # of several sites
    return y, or1, or2, dmse
###############################################################################################################

def Kriging_Main(inputnum,outputnum,using_num,edge,theta,Input_path,Output_path=0,regFunc=regpoly2,corrFunc=corrcubic,mutiplier=0, iOutput=0,iPlot=0):
    ## Initialization
    lob=edge[:,0]
    upb=edge[:,1]

    mE = edge[:, 0].T
    sE = (edge[:, 1] - edge[:, 0]).T
    temp_input = Func_txt_to_list(Input_path)
    m = temp_input.shape[0]

    tS = temp_input[:,0:inputnum]#layer_data[:, 0:inputnum]  # 输入参数集
    tY = temp_input[:,inputnum:(outputnum+inputnum)]

    S = tS[0:using_num,:]
    rS = tS[using_num:,:]
    Y = tY[0:using_num,:]
    rY = tY[using_num:,:]

    ## Model training
    dmodel, perf=dacefit(S, Y, regFunc, corrFunc, theta, lob, upb)

    ## Prediction
    X = rS
    YX = np.zeros([X.shape[0], outputnum])
    if iOutput==1:
        OutputS = open(Output_path, mode='w')
        for i in range(X.shape[0]):
            YX[i,:], _, MSE, _ = predictor(X[i, :], dmodel)
            Mis = abs(YX[i,:] - rY[i,:])
            for j in range(X.shape[1]):
                OutputS.write(str(X[i,j])+'\t')
            for j in range(YX.shape[1]):
                OutputS.write(str(rY[i,j])+'\t')
            for j in range(YX.shape[1]):
                OutputS.write(str(YX[i,j])+'\t')
            for j in range(YX.shape[1]):
                OutputS.write(str(Mis[j])+'\t')
            OutputS.write('\n')
        OutputS.close()
    for i in range(X.shape[0]):
        YX[i,:], _, MSE, _ = predictor(X[i,:], dmodel)

    if iPlot==1:
        if outputnum == 1:
            rY = rY.reshape([-1, 1])
            YX = YX.reshape([-1, 1])
        for i in range(outputnum):
            plt.figure(i)
            ax4 = plt.axes(projection='3d')
            ax4.scatter(rS[:, 0], rS[:, 1], rY[:, i], 'Black')
            ax4.scatter(rS[:, 0], rS[:, 1], YX[:, i], 'red')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax4.set_zlabel('Z')
            plt.show()
    return dmodel