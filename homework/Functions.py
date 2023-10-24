import pandas as pd
import numpy as np

def calculate_statistics(df, annualize_factor=12):
    '''
    Calculates the mean, volatility and sharpe ratio of a dataframe
    Returns a dictionary with the 'mean', 'volatility' and 'sharpe' ratio of each non-date column
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
            res.update({i:res_i})
            res_i={}
    return res

def calculate_statistics_array(data, annualize_factor=12):
    '''
    Calculates the mean, volatility and sharpe ratio of an array
    Returns a dictionary with the 'mean', 'volatility' and 'sharpe' ratio of the array
    '''
    res_i={}
    res_i.update({'mean':np.mean(data)*annualize_factor})
    res_i.update({'volatility':np.std(data)*(annualize_factor**(1/2))})
    res_i.update({'sharpe':res_i['mean']/res_i['volatility']})
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