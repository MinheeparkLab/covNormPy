"""
dealing with various normalization methods.

ICE (Iterative Correction and Eigenvalue Decomposition)
    by M.Imakaev et al

SCN (Sequential Component Normalization)
    by Cournac A et al
    
KR (Knight-Ruiz)
    by Knight PA et al
    
covNorm (coverage based Normalization)
    - distNorm (distance based Normalization)
    by K.Kim and I.Jung

code written by: Han, manhyuk (manhyukhan@kaist.ac.kr)
Korea Advanced Institute of Science and Technology

"""
from os import TMP_MAX
from numpy.random import sample
import pandas as pd
import numpy as np
from scipy import sparse
from scipy import stats
import statsmodels.api as sm
import warnings
import time

def ICE(X, SS=None, max_iter=3000, eps=1e-4, copy=True,
                      norm='l1', verbose=0, output_bias=False,
                      total_counts=None, counts_profile=None):
    """
    Iterative Correction function depends on hiclib.iced (or pastis.iced)
    
    The source code is from https://github.com/hiclib/iced
    
    python >= 2.7 numpy >= 1.16 scipy >= 0.19 sklearn pandas
    
    Parameters
    ----------
    X : ndarray or sparse array (n, n)
        raw interaction frequency matrix

    max_iter : integer, optional, default: 3000
        Maximum number of iteration

    eps : float, optional, default: 1e-4
        the relative increment in the results before declaring convergence.

    copy : boolean, optional, default: True
        If copy is True, the original data is not modified.

    norm : string, optional, default: l1
        If set to "l1", will compute the ICE algorithm of the paper. Else, the
        algorithm is adapted to use the l2 norm, as suggested in the SCN
        paper.

    output_bias : boolean, optional, default: False
        whether to output the bias vector.

    total_counts : float, optional, default: None
        the total number of contact counts that the normalized matrix should
        contain. If set to None, the normalized contact count matrix will be
        such that the total number of contact counts equals the initial number
        of interactions.

    Returns
    -------
    X, (bias) : ndarray (n, n)
        Normalized IF matrix and bias of output_bias is True
    """
    try:
        import iced
    except ImportError:
        warnings.warn('No module iced found\ntry pastis.externals.iced instead')
        import pastis.externals.iced as iced
    except ImportError:
        warnings.warn('No module pastis found\ntry cooltools instead.\nNotice that cooltools scale 0.8')
        from cooltools.lib.numutils import iterative_correction_symmetric
        return iterative_correction_symmetric(X,max_iter=max_iter,tol=eps,verbose=verbose)[0]
    except ImportError:
        raise ImportError('No possible ICE module found. Please install it')
    
    return iced.normalization.ICE_normalization(X, SS=SS, max_iter=max_iter, eps=eps, copy=copy,
                      norm=norm, verbose=verbose, output_bias=output_bias,
                      total_counts=total_counts, counts_profile=counts_profile)
    
def SCN(X, max_iter=300, eps=1e-6, copy=True):
    """
    Parameters
    ----------
    X : ndarray (n, n)
        raw interaction frequency matrix

    max_iter : integer, optional, default: 300
        Maximum number of iteration

    eps : float, optional, default: 1e-6
        the relative increment in the results before declaring convergence.

    copy : boolean, optional, default: True
        If copy is True, the original data is not modified.

    Returns
    -------
    X : ndarray,
        Normalized IF
    """
    try:
        import iced
    except ImportError:
        warnings.warn('No module iced found\ntry pastis.externals.iced instead')
        import pastis.externals.iced as iced
    except ImportError:
        raise ImportError('No possible SCN module found. Please install it')
    
    return iced.normalization.SCN_normalization(X, max_iter=max_iter, eps=eps, copy=copy)
        
def covNorm(X, cov1_thresh=200, cov2_thresh=200, max_covnorm_value=50, total_counts=None,verbose=True):
    """
    covNorm for contact matrix.

    - Contact matrix does not have fragment differences
    
    Parameters
    ----------
    X : ndarray (n, n)
        raw interaction frequency matrix
        
    cov_thresh : int
        Threshold for filtering bins with too low coverage
    
    max_covnorm_value : int
        Outliers are replaced
        
    total_counts : float, optional, default: None
        the total number of contact counts that the normalized matrix should
        contain. If set to None, the normalized contact count matrix will be
        such that the total number of contact counts equals the initial number
        of interactions.
        
    Returns
    -------
    X : ndarray (n,n)
        normalized
    """
    X = X.copy()
    X[np.isnan(X)] = 0
    n = X.shape[0]
    
    mask1 = np.sum(X,axis=1) <= cov1_thresh
    mask2 = np.sum(X,axis=0) <= cov2_thresh
    
    X[mask1 & mask2] = 0
    if total_counts is None: 
        totalCounts = X.sum()
    
    #X = sparse.coo_matrix(X)
    cov1 = np.sum(X,axis=1)
    cov2 = np.sum(X,axis=0)
    
    X = sparse.coo_matrix(X)
    expect = X.data.copy()
    cov1Vec = np.array([cov1[i] for i in X.row])
    cov2Vec = np.array([cov2[i] for i in X.col])
    m = expect.shape[0]
    
    expectFit = expect.copy().reshape(m,1)
    cov1VecFit = cov1Vec.copy().reshape(m,1)
    cov2VecFit = cov2Vec.copy().reshape(m,1)
    
    corrBefore = [stats.pearsonr(cov1Vec,expect)[0],stats.pearsonr(cov2Vec,expect)[0]]
    
    D = np.concatenate([np.ones(m).reshape(m,1),cov1VecFit,cov2VecFit],axis=1)
    print('-- linear regression --')
    result = sm.GLM(expectFit,D,family=sm.families.NegativeBinomial(link=sm.genmod.families.links.log)).fit()
    params = result.params
    if verbose: print(result.summary())
        
    expectCapture = np.round(np.exp(params[0]+params[1]*cov1Vec+params[2]*cov2Vec),4)
    assert expectCapture.shape[0] == m
    X.data = np.round(X.data / expectCapture,4)
    X.data[np.isnan(X.data)] = 0
    X.data[np.isinf(X.data)] = max_covnorm_value
    X.data *= totalCounts/X.data.sum()
    assert X.shape == (n,n)
    ## symmetrisize ##
    Xsym = sparse.coo_matrix((1/2)*(X.toarray() + X.toarray().transpose()))
    
    if verbose:
        cov1 = np.sum(Xsym,axis=1)
        cov2 = np.sum(Xsym,axis=0)
        cov1Vec = np.array([cov1[i,0] for i in Xsym.row])
        cov2Vec = np.array([cov2[0,i] for i in Xsym.col])
        corrAfter = [stats.pearsonr(cov1Vec,Xsym.data)[0],stats.pearsonr(cov2Vec,Xsym.data)[0]]
        print(f"---Pearson Correlation Change---\n-Before : {corrBefore}\n-After : {corrAfter}")
    
    return Xsym.toarray()

def covNormDF(dataFrame, doShuffle=True, cov1_thresh=200,cov2_thresh=200,
            max_covnorm_value=50, sample_ratio=-1, verbose=True):
    """
    covNorm for dataframe (Hi-C data format of I.Jung lab)
    
    code translated from normCoverage.R
    
    Parameters
    ----------
    dataFrame : pd.DataFrame
        dataFrame for Hi-C data
    
    doShuffle : bool_
        if True, shuffle frag1 and frag2
    
    cov1_thresh : int
        Threshold for filtering bins with too low coverage
    
    cov2_thresh : int
        Threshold for filtering bins with too low coverage
    
    max_covnorm_value : int
        Outliers by division with too small a number are replaced by this value
        
    sample_ratio : float
        If not -1 and [0,1], downsample the df rows to be used for fitting by that ratio
        
    Returns
    -------
    params : np.ndarray
        parameters for linear regression
    
    normDataFrame : pd.DataFrame
        normalizaed dataframe
    """
    mask = (dataFrame['cov_frag1'] > cov1_thresh) & (dataFrame['cov_frag2'] > cov2_thresh)
    dataFrame = dataFrame.iloc[list(mask)]
    n = dataFrame.shape[0]
    dataFrame.index = range(n)
    
    dataFrame['rand'] = np.random.rand(n)
    
    if doShuffle:               # time consuming
        for i in range(n):
            if float(dataFrame['rand'][i]) < 0.5:
                tmp = dataFrame['cov_frag1'][i].copy()
                dataFrame['cov_frag1'][i] = dataFrame['cov_frag2'][i].copy()
                dataFrame['cov_frag2'][i] = tmp.copy()
    time.sleep(0.5)
    
    if sample_ratio > 0:
        sample_ratio = max(sample_ratio,1.0)
        dataFrame = dataFrame.sample(frac=sample_ratio)
    
    expect = np.array(dataFrame['freq']).reshape(n,1)
    cov1 = np.array(dataFrame['cov_frag1']).reshape(n,1)
    cov2 = np.array(dataFrame['cov_frag2']).reshape(n,1)
    
    expectFit = expect.copy().reshape(n,1)
    cov1Fit = cov1.copy().reshape(n,1)
    cov2Fit = cov2.copy().reshape(n,1)
    
    #_,X = patsy.dmatrices('expectFit ~ 1 + cov1Fit + cov2Fit',data={'expectFit':expectFit,'cov1Fit':cov1Fit,'cov2Fit':cov2Fit})
    X = np.concatenate([np.ones(n).reshape(n,1),cov1Fit,cov2Fit],axis=1)
    print('-- linear regression --')
    result = sm.GLM(expectFit,X,family=sm.families.NegativeBinomial(link=sm.genmod.families.links.log)).fit()
    params = result.params
    if verbose: print(result.summary())
    
    expectCapture = np.round(np.exp(params[0]+params[1]*cov1+params[2]*cov2),4)
    captureRes = np.round(expect/expectCapture,4)
    captureRes[np.isnan(captureRes)] = 0
    captureRes[np.isinf(captureRes)] = max_covnorm_value
    
    normDataFrame = dataFrame.copy()
    normDataFrame['exp_value_capture'] = expectCapture
    normDataFrame['capture_res'] = captureRes
    
    return params, normDataFrame

def distNorm(X, max_distnorm_value=50, total_counts=None, verbose=False):
    """
    covNorm for contact matrix.

    - Contact matrix does not have fragment differences
    
    Parameters
    ----------
    X : ndarray (n, n)
        raw interaction frequency matrix
    
    max_distnorm_value : int
        Outliers by division with too small a number are replaced by this value
        
    total_counts : float, optional, default: None
        the total number of contact counts that the normalized matrix should
        contain. If set to None, the normalized contact count matrix will be
        such that the total number of contact counts equals the initial number
        of interactions.
        
    Returns
    -------
    X : ndarray (n,n)
        normalized
    """
    X = X.copy()
    X[np.isnan(X)] = 0
    n = X.shape[0]

    if total_counts is None: 
        totalCounts = X.sum()

    dist = lambda x,y: np.abs(x - y)
    
    X = sparse.coo_matrix(X)
    expect = X.data.copy()
    distVec = np.array([dist(X.row[i],X.col[i]) for i in range(len(X.row))])
    m = expect.shape[0]
    
    expectFit = expect.copy().reshape(m,1)
    distVecFit = distVec.copy().reshape(m,1)
    
    corrBefore = stats.pearsonr(distVec,expect)[0]
    
    D = np.concatenate([np.ones(m).reshape(m,1),distVecFit],axis=1)
    print('-- linear regression --')
    result = sm.GLM(expectFit,D,family=sm.families.NegativeBinomial(link=sm.genmod.families.links.log)).fit()
    params = result.params
    if verbose: print(result.summary())
        
    expectCapture = np.round(np.exp(params[0]+params[1]*distVec),4)
    assert expectCapture.shape[0] == m
    X.data = np.round((X.data + np.mean(X.data))/ (expectCapture + np.mean(X.data)),4)
    X.data[np.isnan(X.data)] = 0
    X.data[np.isinf(X.data)] = max_distnorm_value
    X.data *= totalCounts/X.data.sum()
    assert X.shape == (n,n)
    
    corrAfter = stats.pearsonr(distVec,X.data)[0]
    
    if verbose:
        print(f"---Pearson Correlation Change---\n-Before : {corrBefore}\n-After : {corrAfter}")
    
    return X.toarray()

def distNormDF(dataFrame, max_dist = -1, max_distnorm_value = 50, sample_ratio = -1):
    """
    distNorm for dataframe (Hi-C data format of I.Jung lab)
    
    code translated from normDistance.R
    
    Parameters
    ----------
    dataFrame : pd.DataFrame
        dataFrame for Hi-C data
    
    max_dist : int
        If > 0, Rows with larger distance values will be dropped
    
    max_distnorm_value : int
        Outliers by division with too small a number are replaced by this value
        
    sample_ratio : float
        If not -1 and [0,1], downsample the df rows to be used for fitting by that ratio
        
    Returns
    -------
    params : np.ndarray
        parameters for linear regression
    
    normDataFrame : pd.DataFrame
        normalizaed dataframe
    """
    if max_dist > 0:
        mask = dataFrame['dist'] > max_dist
        dataFrame = dataFrame.iloc[list(mask)]
    
    n = dataFrame.shape[0]
    dataFrame.index = range(n)
    
    if sample_ratio > 0:
        sample_ratio = max(sample_ratio,1.0)
        dataFrame = dataFrame.sample(frac=sample_ratio)
    
    expect = np.array(dataFrame['freq']).reshape(n,1)
    dist = np.array(dataFrame['dist']).reshape(n,1)
    
    expectFit = expect.copy().reshape(n,1)
    distFit = dist.copy().reshape(n,1)
    
    X = np.concatenate([np.ones(n).reshape(n,1),distFit],axis=1)
    print('-- linear regression --')
    result = sm.GLM(expectFit,X,family=sm.families.NegativeBinomial(link=sm.genmod.families.links.log)).fit()
    params = result.params
    
    expectCapture = np.round(np.exp(params[0]+params[1]*dist),4)
    distanceRes = np.round((expect + np.mean(expect))/(expectCapture + np.mean(expect)),4)
    distanceRes[np.isnan(distanceRes)] = 0
    distanceRes[np.isinf(distanceRes)] = max_distnorm_value
    
    normDataFrame = dataFrame.copy()
    normDataFrame['exp_value_dist'] = expectCapture
    normDataFrame['dist_res'] = distanceRes
    
    return params, normDataFrame
            