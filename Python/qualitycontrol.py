"""
dealing with the statistical analysis and quanlity control for Normalization results.

References:
    K. Kim and I. Jung, Comput. Struct. Biotechnol. J., 2021, 19, 3149–3159.

code written by: Han, manhyuk (manhyukhan@kaist.ac.kr) 12.18.2021
Korea Advanced Institute of Science and Technology
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy import sparse
from statsmodels.stats.multitest import fdrcorrection as fdr
from distfit import distfit

def checkFreqCovPCC(dataFrame):
    pccBeforeCovNorm1 = stats.pearsonr(dataFrame['cov_frag1'],dataFrame['freq'])
    pccBeforeCovNorm2 = stats.pearsonr(dataFrame['cov_frag2'],dataFrame['freq'])
    
    print(f"Pearson's correlation before normalization :\n fragment 1 {pccBeforeCovNorm1[0]} fragment 2 {pccBeforeCovNorm2[0]}")

    pccAfterCovNorm1 = stats.pearsonr(dataFrame['cov_frag1'],dataFrame['capture_res'])
    pccAfterCovNorm2 = stats.pearsonr(dataFrame['cov_frag2'],dataFrame['capture_res'])
    
    print(f"Pearson's correlation after normalization\n fragment 1 {pccAfterCovNorm1[0]} fragment 2 {pccAfterCovNorm2[0]}")

def checkFreqDistPCC(dataFrame):
    pccBeforedistNorm = stats.pearsonr(dataFrame['dist'],dataFrame['cpature_res'])
    
    print(f"Pearson's correlation before normalization:\n {pccBeforedistNorm[0]}")
    
    pccAfterdistNorm = stats.pearsonr(dataFrame['dist'],dataFrame['dist_res'])
    
    print(f"Pearsons's correlation after normalization\n {pccAfterdistNorm[0]}")

def contactPvalDF(dataFrame, outfilename=None, scope='popular'):
    """
    calculate p-values by using distfit module.
    
    Parameters
    ----------
    dataFrame : pd.DataFrame
        input dataframe (format of I.Jung lab)
        
    outfilename : str, default None
        output file name if want to save
    
    Returns
    -------
    dataFrame : pd.DataFrame
        dataFrame including 'p_result_dist' and 'FDR_dist_res'
            p_result_res : p-value result
            FDR_dist_res : FDR result
    
    """
    assert scope in ['full','popular']
    ## try fitting for low dist_res values ##
    distRes = np.float32(dataFrame['dist_res'])
    distResFilter = distRes[distRes <= 2]
    
    fit = distfit(todf=True,distr=scope)
    fit.fit_transform(distResFilter)
    print(fit.summary)
    
    modelname = fit.model['name']
    print(f"Best fitted model is {modelname}")
    fit.predict(distRes)
    
    result = fit.results['df']
    
    dataFrame['p_result_dist'] = result['P']
    
    fdrResult = fdr(np.array(result['P']))[1]
    dataFrame['FDR_dist_res'] = fdrResult
    
    return dataFrame

def contactPval(X, outfilename=None, scope='popular'):
    """
    calculate p-values and FDR by using distfit
    Matrix form
    
    Parameters
    ----------
    X : ndarray (n, n)
        input contact matrix
    
    outfilename : str, default None
        output file name
        
    Returns
    -------
    dataFrame : pd.DataFrame
        dataFrame having p value and fdr result
        
    """
    assert scope in ['popular', 'full']
    
    X = X.copy()
    X[np.isnan(X)] = 0

    dist = lambda x,y: np.abs(x - y)
    
    dataFrame = pd.DataFrame()
    X = sparse.coo_matrix(X)
    distVec = np.array([dist(X.row[i],X.col[i]) for i in range(len(X.row))])
    
    dataFrame['dist_res'] = distVec
    
    dataFrame = contactPvalDF(dataFrame, outfilename=outfilename, scope=scope)
    
    return dataFrame

def plotCovNormResult(dataFrame, binsize=100, pct_quantile=0.09, outputname='plot_coverage_norm_result.png'):
    pass


