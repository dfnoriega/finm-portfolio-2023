import pandas as pd
import numpy as np
from sympy import Matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_statistics(df, annualize_factor=12, VaR=0.05, CVaR=0.05):
    '''
    Calculates the mean, volatility, sharpe, skewness, kurtosis, VaR and CVaR of a dataframe
    Returns a dataframe with values for each asset
    '''
    res={}
    res_i={}
    for i in df.columns:
        if df[i].dtype=='<M8[ns]':
            pass
        else:
            res_i.update({'mean':np.mean(df[i])*annualize_factor})
            res_i.update({'volatility':np.std(df[i])*(annualize_factor**(1/2))})
            res_i.update({'sharpe':res_i['mean']/res_i['volatility']})
            res_i.update({'skewness':skew(df[i])})
            res_i.update({'kurtosis':kurtosis(df[i])})
            res_i.update({'VaR':df[i].quantile(VaR)})
            res_i.update({'CVaR':df[i][df[i]<df[i].quantile(CVaR)].mean()})
            # res_i.update({'Max_Drawdown':maxDrawD(df[i])})
            res.update({i:res_i})
            res_i={}
    return pd.DataFrame(res)

def calculate_statistics_array(data, annualize_factor=12, VaR=0.05, CVaR=0.05):
    '''
    Calculates the mean, volatility and sharpe ratio of an array
    Returns a dictionary with the 'mean', 'volatility' and 'sharpe' ratio of the array
    '''
    res_i={}
    res_i.update({'mean':np.mean(data)*annualize_factor})
    res_i.update({'volatility':np.std(data)*(annualize_factor**(1/2))})
    res_i.update({'sharpe':res_i['mean']/res_i['volatility']})
    res_i.update({'skewness':skew(data)})
    res_i.update({'kurtosis':kurtosis(data)})
    df=pd.DataFrame(data)
    res_i.update({'VaR':df.quantile(VaR)})
    res_i.update({'CVaR':df[df<df.quantile(CVaR)].mean()})
    # res_i.update({'Max_Drawdown':maxDrawD(df[i])})
    return res_i

def tangency_portfolio(df):
    '''
    Calculates the weights of the tangency portfolio
    Inputs: dataframe with column 0 being the date and the rest ([1:]) being the assets
    Make sure df's first column (0) is the date, or anything that is not an asset
    '''
    stats=calculate_statistics(df)
    assets=len(df.columns[1:])
    mdf=Matrix(df.iloc[:,1:].cov())
    vect1=Matrix([1]*assets)
    mean=[]
    for i in stats:
        mean.append(stats[i]['mean'])
    vectmean=Matrix(mean)
    sigma_inv=mdf.inv()
    wt=(1/((vect1.T@sigma_inv@vectmean)[0,0]))*(sigma_inv@vectmean)

    tickers=[]
    for i in stats:
        tickers.append(i)
    tan_port=pd.DataFrame()
    tan_port['tickers']=tickers
    tan_port['Tangent Weights']=0
    for i in range(len(tan_port)):
        tan_port.loc[i,'Tangent Weights']=wt[i]
    
    tan_port.set_index('tickers', inplace=True,drop=True)

    return tan_port

def correlation_heatmap(df):
    '''
    Plots a heatmap of the correlation matrix of a dataframe [1:]
    '''
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.iloc[:,1:].corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)

def calculate_market_statistics(df,regressor, annualize_factor=12):
    '''
    Calculates the alpha, beta, treynor ratio and information ratio of a dataframe for all non-date columns
    Regressor is benchmark: e.g., SPY, Market
    '''
    res={}
    res_i={}
    for i in df.columns:
        if df[i].dtype=='<M8[ns]':
            pass
        else:
            model=LinearRegression()
            model.fit(np.array(regressor).reshape(-1,1),df[i])
            alpha=model.intercept_*annualize_factor
            beta=model.coef_[0]
            res_i.update({'alpha':alpha})
            res_i.update({'market_beta':beta})
            res_i.update({'treynor_ratio':np.mean(df[i])*annualize_factor/beta})

            residuals=np.array(df[i])-model.predict(np.array(regressor).reshape(-1,1))
            res_i.update({'information_ratio':alpha/(np.std(residuals)*annualize_factor**(1/2))})

            res.update({i:res_i})
            res_i={}
    return pd.DataFrame(res).transpose()

def run_regression(df,regressors,annualize_factor=12):
    '''
    Runs a regression for all non-date columns in a dataframe
    Regressors is a dataframe with the regressors as columns
    Returns a dataframe with the alpha and betas for each asset
    '''
    res={}
    res_i={}
    for i in df.columns:
        if df[i].dtype=='<M8[ns]':
            pass
        else:
            model=LinearRegression()
            model.fit(regressors,df[i])
            alpha=model.intercept_*annualize_factor
            betas=model.coef_
            res_i.update({'alpha':alpha})
            for ii in range(len(betas)):
                res_i.update({'beta_'+str(regressors.columns[ii]):betas[ii]})
            res.update({i:res_i})
            res_i={}
    
    return pd.DataFrame(res).transpose()

