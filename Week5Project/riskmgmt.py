import numpy as np
import pandas as pd
import math
from scipy.stats import norm,t,kurtosis
from scipy.optimize import minimize

# near_psd
def near_psd(a,epsilon=0):
    n= a.shape[0]
    invSD = None
    out = a.copy()
    if (np.diag(out)==1).sum() != n:
        invSD = np.diag(1/np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    vals, vecs = np.linalg.eigh(out)
    vals[vals<epsilon]=0
    T = 1/(vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if invSD is not None:
        invSD = np.diag(1/np.diag(invSD))
        out = invSD @ out @ invSD
    return out

# The first projection for Higham method which assume that the weight matrix is diagonal
def pu(x):
    n = x.shape[0]
    x_pu = x.copy()
    for i in range(n):
        for j in range(n):
            if i==j:
                x_pu[i][j]=1
    return x_pu

# The second projection for Higham method
def ps(x,w=None):
    n = x.shape[0]
    if w != None:
        w_diag = np.diag(w)
    else:
        w_diag = np.diag(np.ones(n))
    x_w = np.sqrt(w_diag) @ x @ np.sqrt(w_diag)
    vals, vecs = np.linalg.eigh(x_w)
    vals[vals<1e-8]=0
    l = np.diag(vals)
    x_pos = vecs @ l @ vecs.T
    w_inv = np.linalg.inv(np.sqrt(w_diag))
    out = w_inv @ x_pos @ w_inv
    return out

# Frobenius Norm
def fnorm(x):
    n = x.shape[0]
    result = 0
    for i in range(n):
        for j in range(n):
            result += x[i][j] ** 2
    return result

# higham
def higham(a,gamma0=np.inf,K=100,tol=1e-08):
    delta_s = [0]
    gamma = [gamma0]
    Y = [a]
    for k in range(1,K+1):
        R_k = Y[k-1] - delta_s[k-1]
        X_k = ps(R_k)
        delta_s_k = X_k - R_k
        delta_s.append(delta_s_k)
        Y_k = pu(X_k)
        Y.append(Y_k)
        gamma_k = fnorm(Y_k-a)
        gamma.append(gamma_k)
        if gamma_k -gamma[k-1] < tol:
            vals = np.linalg.eigh(Y_k)[0]
            if vals.min() >= 1e-8:
                break
            else:
                continue
    return Y[-1]

# Check if a matrix is PSD
def is_psd(matrix):
    eigenvalues = np.linalg.eigh(matrix)[0]
    return np.all(eigenvalues >= -1e-8)

# chol_psd
def chol_psd(a):
    n= a.shape[0]
    root = np.zeros((n,n))
    for j in range(n):
        s=0
        if j>0:
            s = root[j,:j].T @ root[j,:j]
        temp = a[j,j] - s
        if temp <= 0 and temp >= -1e-8:
            temp =0
        root[j,j] = math.sqrt(temp)
        if root[j,j] == 0:
            root[j+1:n,j] = 0
        else:
            ir = 1/root[j,j]
            for i in range(j+1,n):
                s = root[i,:j].T @ root[j,:j]
                root[i,j] = (a[i,j]-s)*ir
    return root

#normal simulation
def normal_sim(a,nsim,seed,means=[],fixmethod=near_psd):
    eigval_min = np.linalg.eigh(a)[0].min()
    if eigval_min < 1e-08:
        a = fixmethod(a)
    l = chol_psd(a)
    m = l.shape[0]
    np.random.seed(seed)
    z = np.random.normal(size=(m,nsim))
    X = (l @ z).T
    if means.size != 0:
        if means.size != m:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            X[:,i] = X[:,i] + means[i]
    return X

def pca_vecs(cov):
    eigvalues, eigvector = np.linalg.eigh(cov)
    vals = np.flip(eigvalues)
    vecs = np.flip(eigvector,axis=1)
    posv_ind = np.where(vals >= 1e-8)[0]
    vals = vals[posv_ind]
    vecs = vecs[:,posv_ind]
    vals = np.real(vals)
    return vals,vecs

def vals_pct(vals,vecs,pct):
    tv = vals.sum()
    for k in range(len(vals)):
        explained = vals[:k+1].sum()/tv
        if explained >= pct:
            break
    return vals[:k+1],vecs[:,:k+1]

# pca simulation
def pca_sim(a,nsim,seed,means=[],pct=None):
    vals,vecs = pca_vecs(a)
    if pct != None:
        vals,vecs = vals_pct(vals,vecs,pct)
    B = vecs @ np.diag(np.sqrt(vals))
    m = vals.size
    np.random.seed(seed)
    r = np.random.normal(size=(m,nsim))
    out = (B @ r).T
    if means.size != 0:
        if means.size != out.shape[1]:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            out[:,i] = out[:,i] + means[i]
    return out

# Generate weight
def weights_gen(lamda,t):
    tw = 0
    w = np.zeros(t)
    for i in range(t):
        w[i] = (1-lamda)*lamda ** (t-i-1)
        tw += w[i]
    for i in range(t):
        w[i] = w[i]/tw
    return w

# EW cov + var
def w_cov(df,lamda):
    n = df.shape[1]
    t = df.shape[0]
    w = weights_gen(lamda,t)
    means = np.array(df.mean())
    xhat = df.copy()
    for i in range(n):
        xhat.iloc[:,i]=xhat.iloc[:,i]-means[i]
    cov = xhat.multiply(w,axis=0).T @ xhat
    return cov

# Pearson corr + var
def pcov(df):
    vars =df.var()
    std = np.sqrt(vars)
    corr = np.corrcoef(df,rowvar=False)
    cov = np.diag(std) @ corr @ np.diag(std)
    return cov

# Pearson correlation and EW variance
def pcor_ewvar(df,lamda):
    w_var = np.diag(w_cov(df,lamda))
    w_std = np.sqrt(w_var)
    corr = np.corrcoef(df,rowvar=False)
    cov = np.diag(w_std) @ corr @ np.diag(w_std)
    return cov

# EW corr + Var
def wcor_var(df,lamda):
    wcov = w_cov(df,lamda)
    w_var = np.diag(w_cov(df,lamda))
    w_std = np.sqrt(w_var)
    w_corr = np.diag(1/w_std) @ wcov @ np.diag(1/w_std)
    vars =df.var()
    std = np.sqrt(vars)
    cov = np.diag(std) @ w_corr @ np.diag(std)
    return cov

# Implement return_calculate
def return_calculate(df,method):
    if df.columns[0]=='Date':
        ind = df.columns[1:]
        datesig = True
    else:
        ind = df.columns
        datesig = False
    p = df.loc[:,ind]
    n = p.shape[1]
    t = p.shape[0]
    p2 = np.zeros((t-1,n))
    for i in range(t-1):
        for j in range(n):
            p2[i,j]=p.iloc[i+1,j]/p.iloc[i,j]
    if method.upper()== "DISCRETE":
        p2 = p2 -1
    elif  method.upper()== "LOG":
        p2 = np.log(p2)
    else:
        raise Exception("Method be either discrete or log")
    out = pd.DataFrame(data=p2,columns=ind)
    if datesig == True:
        out.insert(0,'Date',np.array(df.loc[1:,'Date']))
    return out

# Get the historical prices and holdings of a portfolio
def port_cal(port,stockdata,portdata,method="discrete"):
    if port == "All":
        port_prices = stockdata.loc[:,portdata['Stock']]
        port_info = portdata
    else:
        port_info = portdata[portdata['Portfolio']==port]
        port_prices = stockdata.loc[:,port_info['Stock']]

    cur_price = port_prices.iloc[-1,:]
    cur_value = (cur_price * np.array(port_info['Holding'])).sum()

    r = return_calculate(port_prices,method=method)
    return r,cur_price,cur_value,port_info

def VaR(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    return -v

def ES(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    es = a[a<=v].mean()
    return -es

def simulate_error(errorModel, errors, u):
    sim_val = errorModel.ppf(u)
    return sim_val

# Fit with normal distribution
def fit_norm(x,n=10000):
    ysim = np.random.normal(x.mean(),x.std(),n)
    return ysim

def fit_norm_error(x):
    m = x.mean()
    s = x.std()
    errorModel = norm(m,s)
    errors = x - m
    u = errorModel.cdf(x)
    return simulate_error(errorModel, errors, u)

# Fit with generalized T distribution
def fit_general_t(x,n=10000):
    def t_fit(vals,r):
        nu = vals[0]
        miu = vals[1]
        s = vals[2]
        ll = t.logpdf(r,df=nu,loc=miu,scale=s).sum()
        return -ll
    start_m = x.mean()
    start_nu = 6.0/kurtosis(x) + 4
    start_s = math.sqrt(x.var()*start_nu/(start_nu-2))
    ll_t_res = minimize(t_fit,[start_nu,start_m,start_s],args=x,
    constraints=({'type':'ineq','fun': lambda vals: vals[0]-2.001},{'type':'ineq','fun': lambda vals: vals[2]-1e-6}))
    print(ll_t_res.message)
    nu,miu,s = ll_t_res.x[0],ll_t_res.x[1],ll_t_res.x[2]
    ysim = t.rvs(df=nu,loc=miu,scale=s,size=n)
    return ysim

def fit_general_t_error(x):
    def t_fit(vals,r):
        nu = vals[0]
        miu = vals[1]
        s = vals[2]
        ll = t.logpdf(r,df=nu,loc=miu,scale=s).sum()
        return -ll
    ll_t_res = minimize(t_fit,[2,0,x.std()],args=x,
    constraints=({'type':'ineq','fun': lambda vals: vals[0]-2},{'type':'ineq','fun': lambda vals: vals[2]}))
    nu,miu,s = ll_t_res.x[0],ll_t_res.x[1],ll_t_res.x[2]
    errorModel = t(df=nu,loc=miu,scale=s)
    errors = x - miu
    u = errorModel.cdf(x)
    return simulate_error(errorModel, errors, u)

def delta_norm(port,stockdata,portdata,alpha=0.05):
    r,cur_price,cur_value,portinfo = port_cal(port, stockdata, portdata)
    sigma = w_cov(r, 0.94)
    n= cur_price.size
    delta = np.zeros(n)
    for i in range(n):
        delta[i] = portinfo.iloc[i,2]*cur_price[i]/cur_value
    delta = pd.DataFrame(delta,index=sigma.index)
    scaler = np.sqrt(delta.T @ sigma @ delta)
    VaR = -cur_value*norm.ppf(alpha)*scaler
    VaR_pct = -norm.ppf(alpha)*scaler
    return VaR.iloc[0,0],VaR_pct.iloc[0,0]

# Historical Simulation
def sim_his(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = port_cal(port, stockdata, portdata)
    np.random.seed(seed)
    r_sim = r.sample(nsim,replace=True)
    p_new = (1+r_sim).mul(cur_price)
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = VaR(profit, alpha)
    es = ES(profit, alpha)
    return var,es

# Monte Carlo Simulation
def sim_mc(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = port_cal(port, stockdata, portdata)
    r_h = r.sub(r.mean(),axis=1)
    sigma= w_cov(r_h, 0.94)
    r_sim = pca_sim(sigma, nsim, seed, means=r.mean(), pct=None)
    r_sim =pd.DataFrame(r_sim,columns=r.columns)
    p_new = (r_sim+1).mul(cur_price)
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = VaR(profit, alpha)
    es = ES(profit, alpha)
    return var,es

def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars_ = prices.columns
    nVars = len(vars_)
    vars_ = [var for var in vars_ if var != dateColumn]
    if nVars == len(vars_):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars_}")
    nVars = nVars - 1
    p = prices[vars_].to_numpy()
    n, m = p.shape
    p2 = np.empty((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices[dateColumn].iloc[1:n].to_numpy()
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars_[i]] = p2[:, i]
    return out


def get_portfolio_price(portfolio, prices, portfolio_code, Delta=False):
    if portfolio_code == "All":
        assets = portfolio.drop('Portfolio',axis=1)
        assets = assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        assets = portfolio[portfolio["Portfolio"] == portfolio_code]        
    stock_codes = list(assets["Stock"])
    assets_prices = pd.concat([prices["Date"], prices[stock_codes]], axis=1)  
    current_price = np.dot(prices[assets["Stock"]].tail(1), assets["Holding"])
    holdings = assets["Holding"]    
    if Delta == True:
        asset_values = assets["Holding"].values.reshape(-1, 1) * prices[assets["Stock"]].tail(1).T.values
        delta = asset_values / current_price    
        return current_price, assets_prices, delta   
    return current_price, assets_prices, holdings

def multivariate_normal_simulation(covariance_matrix, n_samples, method='direct', mean = 0, explained_variance=1.0):
    if method == 'direct':      
        L = chol_psd(covariance_matrix)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))       
        samples = np.transpose(np.dot(L, normal_samples) + mean)        
        return samples 
    elif method == 'pca':
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = eigenvalues > 1e-8
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        if explained_variance == 1.0:
            explained_variance = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]
        n_components = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= explained_variance)[0][0] + 1
        eigenvectors = eigenvectors[:,:n_components]
        eigenvalues = eigenvalues[:n_components]
        normal_samples = np.random.normal(size=(n_components, n_samples))
        B = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
        samples = np.transpose(np.dot(B, normal_samples))      
        return samples
    
def MLE_T(params, returns):
    negLL = -1 * np.sum(t.logpdf(returns, df=params[0], loc=params[1], scale=params[2]))
    return(negLL)

def Fitting_t_MLE(returns):
    constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, {"type":"ineq", "fun":lambda x: x[2]})
    returns_t = minimize(MLE_T, x0=[10, np.mean(returns), np.std(returns)], args=returns, constraints=constraints)
    df, loc, scale = returns_t.x[0], returns_t.x[1], returns_t.x[2]
    return df, loc, scale

def es(data, var):
  return -np.mean(data[data <= -var])

def var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)

