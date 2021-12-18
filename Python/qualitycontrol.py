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

def checkFreqCovPCC(dataFrame):
    pccBeforeCovNorm1 = stats.pearsonr(dataFrame['cov_frag1'],dataFrame['freq'])
    pccBeforeCovNorm2 = stats.pearsonr(dataFrame['cov_frag2'],dataFrame['freq'])
    
    print(f"Pearson's correlation before normalization :\n fragment 1 {pccBeforeCovNorm1} fragment 2 {pccBeforeCovNorm2}")

    pccAfterCovNorm1 = stats.pearsonr(dataFrame['cov_frag1'],dataFrame['capture_res'])
    pccAfterCovNorm2 = stats.pearsonr(dataFrame['cov_frag2'],dataFrame['capture_res'])
    
    print(f"Pearson's correlation after normalization\n fragment 1 {pccAfterCovNorm1} fragment 2 {pccAfterCovNorm2}")


def plotCovNormResult(dataFrame, binsize=100, pct_quantile=0.09, outputname='plot_coverage_norm_result.png'):
    pass