"""
Python porting version of covNorm

Reference:
    K. Kim and I. Jung, Comput. Struct. Biotechnol. J., 2021, 19, 3149–3159.

code written by: Han, manhyuk (manhyukhan@kaist.ac.kr)
Korea Advanced Institute of Science and Technology
"""

import numpy as np
import pandas as pd
import gzip
from . import normalization
from . import qualitycontrol

__version__ = "1.1.1"

def loadGz(filename):
    """
    load .gz file format
    
    Parameters
    ----------
    filename : str
        filename (with or without flag)
    
    Returns
    -------
    dataFrame: pd.DataFrame
        dataFrame from .gz file
    
    """
    if type(filename)!= str: raise TypeError
    if '.gz' not in filename: filename = filename + '.gz'
    dataFrame = pd.read_csv(filename, compression='gzip', header=0, sep='\t', quotechar='"')

    # QC
    with gzip.open(filename,'rt',encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
    
    assert list(dataFrame.keys()) == header
    return dataFrame

def filterInputDF(dataFrame, check_trans_values = True, remove_zero_contact = True, self_ligation_distance = 15000):
    """
    filter input dataframe
    
    Parameters
    ----------
    dataFrame : pd.DataFrame
        input dataframe.
        Hi-C data format in KAIST I.Jung lab
        
    check_trans_values : bool_
        row containing trans interactions dropped
    
    remove_zero_contacts : bool_
        discard rows containing zero frequencies
        
    self_ligation_distance : int
        distance between two fragments regarded as self-ligation.
        
    Returns
    -------
    dataFrameFiltered : pd.DataFrame
        filtered dataframe
    
    """
    mask = [True for i in range(dataFrame.shape[0])]
    
    if check_trans_values:
        mask = dataFrame['frag1'].str.split('.',expand=True)[0] == dataFrame['frag2'].str.split('.',expand=True)[0]
    
    if remove_zero_contact:
        mask = mask & (dataFrame['freq'] > 0)
    
    if self_ligation_distance:
        mask = mask & (dataFrame['dist'] > self_ligation_distance)
    
    dataFrameFiltered = dataFrame[mask]
    
    return dataFrameFiltered

def toMatrix(dataFrame, chrLength=False, resolution=False, freqCol='freq'):
    """
    Convert dataFrame (I.Jung lab format) into contact matrix
    
    Parameters
    ----------
    dataFrame : pd.DataFrame
        dataFrame want to convert
    
    chrLength : int (or bool_ only in case of default)
        chromosome length. default infer from data
    
    resolution : int (or bool_ only in case of default)
        resolution (bin size) of the matrix. default infer from data
        
    freqCol : str
        key name considered as contact, default is 'freq'
    
    Returns
    -------
    X : ndarray (n, n)
        contact matrix
    
    """
    dataFrame = dataFrame.copy()
    
    if not (chrLength or resolution):
        if not chrLength:
            chrLength = max(max(np.int64(dataFrame['frag1'].str.split('.',expand=True)[1])),
                            max(np.int64(dataFrame['frag2'].str.split('.',expand=True)[1])))
            
        if not resolution:
            resolution = int(dataFrame['frag1'][0].split('.')[2]) - int(dataFrame['frag1'][0].split('.')[1])
        
    n = int(chrLength/resolution +1)
    
    X = np.zeros((n,n))
    
    m = dataFrame.shape[0]
    dataFrame.index = range(m)
    totalFreq = 0
    for i in range(m):
        bin1 = int(dataFrame['frag1'][i].split('.')[1])/resolution
        bin2 = int(dataFrame['frag2'][i].split('.')[1])/resolution
        freq = float(dataFrame[freqCol][i])
        totalFreq += freq
        
        X[int(bin1),int(bin2)] += freq
        X[int(bin2),int(bin1)] += freq

    assert (X.sum() - 2*totalFreq) < 1e-5
    
    return X

