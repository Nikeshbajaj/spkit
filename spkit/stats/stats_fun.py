"""
Statistical Utalities
--------------------
Author @ Nikesh Bajaj
updated on Date: 16 March 2023, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk | nikesh.bajaj@qmul.ac.uk

"""



import numpy as np
import pandas as pd
import scipy, copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy import stats as scipystats

from ..utils import bcolors
from ..core.information_theory import entropy

def quick_stats(x,show=True,return_table=True,name='x'):
    r"""Quick Statistics, excluding np.nan values


    Parameters
    ----------
    x: 1d-array
      - array of numbers

    show: bool, default=True
      - show table as pandas dataframe

    return_table: bool, deafult=True
      - return Table
    
    name: str, default='x'
      - name of the column in table

    Returns
    -------
    Sdf: pd DataFrame
       - Table of statistics


    See Also
    --------
    get_stats, outliers

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import spkit as sp
    >>> x = np.random.randn(1000)
    >>> df = sp.stats.quick_stats(x)
                n      mean        sd    median       min       max      q25      q75  \
    x  1000.0  0.003649  1.022257  0.000009 -3.486239  2.938597 -0.71816  0.66914   
        iqr       kur       skw  
    x  1.3873  0.029687  0.013038 
    """
    xi = x[~np.isnan(x)]
    if len(xi):
        names =['n','mean','sd','median','min','max','q25','q75','iqr','kur','skw']
        S  = [len(xi),np.mean(xi),np.std(xi),np.median(xi),np.min(xi),np.max(xi),
              np.nanquantile(x,0.25),np.nanquantile(x,0.75),scipystats.iqr(x[~np.isnan(x)]),
              scipystats.kurtosis(x,nan_policy='omit'),scipystats.skew(x[~np.isnan(x)])]
        Sdf = pd.DataFrame(S,index=names,columns=[name]).T

        if show:
            try:
                from IPython import display
                display(Sdf)
            except:
                print(Sdf)
        if return_table: return Sdf
    return None

def get_stats(x,detail_level=1,name='x',esp=1e-5,add_esp=False):
    '''Statistics of a given sequence x, excluding NaN values
    
    returns stats and names of statistics measures
    
    Parameters
    ----------
    x: 1d-array
      - array of numbers

    detail_level: int {1,2,3}
      -  level of details
      - For detail_level=1, 6 values:
            - ['mean','sd','median','min','max','n']
      - For detail_level=2: 11 values:
            - ['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw']
      - For detail_level=3: 13 values:
            - ['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    
    name: str, default='x'
      - name of the column in table

    Returns
    -------
    S: 1d-array of values
    
    stats_names: list of str,
       - names of stats

    Sdf: pd DataFrame
       - Table of statistics
    
    
    See Also
    --------
    quick_stats, outliers


    Examples
    --------
    >>> #sp.stats.get_stats
    >>> import numpy as np
    >>> import pandas as pd
    >>> import spkit as sp
    >>> x = np.random.randn(1000)
    >>> S,names,df = sp.stats.get_stats(x,detail_level=1)
    >>> print(df)
           mean        sd  median       min       max       n
    x -0.029575  1.016306 -0.0001 -3.522644  2.942269  1000.0
    >>> S,names,df = sp.stats.get_stats(x,detail_level=3)
    >>> print(df)
            mean        sd  median       min       max       n       q25       q75  \
    x -0.029575  1.016306 -0.0001 -3.522644  2.942269  1000.0 -0.726123  0.640879   
            iqr       kur       skw     gmean   entropy  
    x  1.367002 -0.017887 -0.007691  0.530046  3.945704  
    '''
    stats_names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    #names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    
    if isinstance(x,int) or isinstance(x,float): x =  [x]
    if isinstance(x,list):x = np.array(x)
    assert len(x.shape)==1
    #logsum = self.get_exp_log_sum(x)

    if add_esp: x = x+esp
    mn = np.nanmean(x)
    sd = np.nanstd(x)
    md = np.nanmedian(x)
    min0 = np.nanmin(x)
    max0 = np.nanmax(x)

    n = len(x) - sum(np.isnan(x))

    xi_nan = x[~np.isnan(x)]
    if detail_level==1:
        S = np.r_[mn,sd,md,min0,max0,n]
        Sdf = pd.DataFrame(S,index=stats_names[:6],columns=[name]).T
        return S, stats_names[:6], Sdf

    q25 = np.nanquantile(x,0.25)
    q75 = np.nanquantile(x,0.75)
    iqr = scipystats.iqr(xi_nan)
    kur = scipystats.kurtosis(x,nan_policy='omit')
    skw = scipystats.skew(xi_nan)
    
    if detail_level==2:
        S =np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw]
        Sdf = pd.DataFrame(S,index=stats_names[:11],columns=[name]).T
        return S, stats_names[:11], Sdf

    gmn = scipystats.gmean(np.abs(xi_nan))

    entrp_x = entropy(xi_nan)

    S = np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw,gmn,entrp_x]
    Sdf = pd.DataFrame(S,index=stats_names,columns=[name]).T
    return S, stats_names, Sdf

def outliers(x, method='iqr',k=1.5, include_lower=True, include_upper=True,return_thr = False):
    r"""Statistical Outliers

    This function computes lower and upper limits beyond which all the point are assumed to be outliers

    IQR - Interquartile Range

    .. math ::

        l_t =  Q3 + k \times (Q3-Q1)

        u_t =  Q3 - k \times (Q3-Q1)

    where :math:`k=1.5` and Q1 is first quartile, Q3 and 3rd Quartile


    Standard Deviation

    .. math ::

        l_t =  k \times SD(x)

        u_t =  - k \times SD(x)


    where :math:`k=1.5` and :math:`SD(\cdot)` is Standard Deviation   

    Parameters
    ----------
    x: 1d array or list
      -  if x included NaNs, they are excluded

    method: str {'iqr','sd'}
      - method to compute lower/upper limits according to above equations

    k: scalar, default k=1.5
      - used as per eqaution

    include_lower: bool, default=True
      -  if False, lower threshold is excluded
    
    include_upper: bool, default=True
      - if False, upper threshold is excluded
    
    return_thr: bool,default = False
      - if True, lower and upper thresholds are returnes

    Returns
    -------
    x_outlr: outliers indentified from x
    idx : indices 
      - indices of the outliers in x, after removing Nans 
      - indices are for xi, xi = x[~np.isnan(x)]
    idx_bin: bool array
      - indices of outliers

    (lt,ut) : tupple, 
       - lower and upper limit
       - returns only if `return_thr` is True


    Examples
    --------
    #sp.stats.outliers
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    np.random.seed(1)
    x = np.random.randn(1000)
    t = np.arange(len(x))
    np.random.seed(None)

    x_outlr1, idx1, _, (lt1,ut1) = sp.stats.outliers(x,method='iqr',return_thr=True)
    x_outlr2, idx2, _, (lt2,ut2) = sp.stats.outliers(x,method='sd',k=2,return_thr=True)

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(t,x,'o',color='C0',alpha=0.8)
    plt.plot(t[idx1],x[idx1],'o',color='C3')
    plt.axhline(lt1,color='k',ls='--',lw=1)
    plt.axhline(ut1,color='k',ls='--',lw=1)
    plt.title('Outliers using  IQR')
    plt.ylabel('x')
    plt.subplot(122)
    plt.plot(t,x,'o',color='C0',alpha=0.8)
    plt.plot(t[idx2],x[idx2],'o',color='C3')
    plt.axhline(lt2,color='k',ls='--',lw=1)
    plt.axhline(ut2,color='k',ls='--',lw=1)
    plt.ylabel('x')
    plt.title('Outliers using  SD')
    plt.show()

    """
    #Atleast one of limit should be true
    assert (include_upper+include_lower)
    xi = x.copy()
    if np.sum(np.isnan(x)):
        xi = xi[~np.isnan(x)]
    if method =='iqr':
        q1 = np.nanquantile(xi,0.25)
        q3 = np.nanquantile(xi,0.75)
        ut = q3 + k*(q3-q1)
        lt = q1 - k*(q3-q1)
    elif method =='sd':
        sd = np.nanstd(xi)
        ut = k*sd
        lt = -k*sd
    else:
        #print("Undefine method, choose 'iqr' or 'sd'")
        raise NameError("Undefine method, choose 'iqr' or 'sd'")

    if not(include_lower): lt = -np.inf
    if not(include_upper): ut = -np.inf

    idx_bin = (xi>=ut) | (xi<=lt)
    idx = np.where(idx_bin)[0]
    x_outlr = xi[idx] if len(idx) else []
    if return_thr:
        return x_outlr, idx, idx_bin, (lt,ut)
    return x_outlr, idx, idx_bin

def _getSS(p,thr):
    ss = ('','')
    if p<thr:
        ss = (bcolors.BGreen,bcolors.BEND)
    elif p<=0.1:
        ss = (bcolors.BGrey,bcolors.BEND)
    return ss

def test_2groups(x1,x2,paired=True,alpha=0.05,pre_tests=True,effect_size=True,tval=False,notes=True,
                    print_round=4,title=None,printthr=1,plots=True,figsize=(5,4),return_all=False,):

    r"""Test two groups

    Parameters
    ----------
    x1: 1d-array
    x2: 1d-array

    paired: bool, default=True
      - if True, x1 and x2 are assumed to be paired, and 
        paired tests are applied
    
    alpha: scalar [0,1], default=0.05
      - alpha level,
      - threshold on p-value for passing/failing declaration
    
    pre_tests: bool, default=True
      - if True pre-tests, Shapiro, and Levene results are shown too
      - Shapiro: Normality
      - Levene: Homogeneity of Variance only for Unpaired

    effect_size: bool, default=True
      - To show effect size (mean difference)  and Cohen's D
 
    tval: bool, default=False
      - if True, all the statisitics (such as t-stats) are also shown

    notes: bool, defualt=True
      - to print notes along the pre-tests about interpretation of p-value
    
    print_round: int, default=4
      - rounding off all the numbers to decimal points
      - print_round=4 means upto 4 decimal points
      - print_round=-1 means all the decimal points available

    title: str, default=None
      - if passed as str, used as heading with "Final Test"
      - useful when running many tests
    
    printthr: scalar [0,1], deafult=1
      - threhold on p-value to display the results of final test
      - if p-value of final test is >printthr then 'final test' results are not printed
      - default=1 to always print the results of final test

    plots: bool, default=True
      - if False then plots are avoided
     
    figsize: figsize default=(5,4)
      - for paired, one plot figsize is used as it is.
      - for unpaired, two plots, width is doubled
      
    return_all: bool, default=False
      - if True, two tables of all the results are returned

    Returns
    -------
    tPass: bool
       - True, if any one of the final test was passed (i.e., p-value < alpha)
       - False means, none of the final test was passed

    (df_tests, df_esize): pd.DataFrames
       - df_tests: Table of all the tests
       - df_esize: table of effect size


    References
    ----------
    * Student's t-test :  https://en.wikipedia.org/wiki/Student%27s_t-test
    * Wilcoxon signed-rank test: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    * Shapiro–Wilk test: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
    * Levene's test: https://en.wikipedia.org/wiki/Levene%27s_test
    * Effect Size : https://en.wikipedia.org/wiki/Effect_size

    See Also
    --------
    test_groups

    Notes
    -----
    Check example with notebook for better view of the output

    Examples
    --------
    >>> #sp.stats.test_2groups
    >>> #Example 1: Paired
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> x1 = np.random.randn(100)
    >>> x2 = np.random.randn(100)+0.2
    >>> tPass,(df1,df2) = sp.stats.test_2groups(x1,x2,paired=True,alpha=0.05,tval=True,return_all=True)
    >>> print(df1)
              p-value        stats
    shapiro  0.407988     0.986549
    t-test   0.019631    -2.371901
    wilcox   0.028014  1886.000000
    >>> print(df2)
                 mean_diff   CohensD
    effect_size  -0.292212 -0.319897
    
    >>> #sp.stats.test_2groups
    >>> #Example 2: Unpaired
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> x1 = np.random.randn(10)
    >>> x2 = np.random.randn(11)+0
    >>> tPass,(df1,df2) = sp.stats.test_2groups(x1,x2,paired=False,alpha=0.05,tval=True,return_all=True)
    >>> print(df1)
                 p-value     stats
    shapiro_x1  0.744053  0.956390
    shapiro_x2  0.929865  0.974728
    levene      0.579284  0.318210
    t-test      0.757153  0.313718
    ranksum     0.724771  0.352089
    >>> print(df2)
                 mean_diff   CohensD
    effect_size   0.157087  0.137073
    """

    n = n1 = len(x1)
    n2 = len(x2)
    pre_tests_results = ''
    test_results = {}
    effect_results = {}
    if paired:
        assert n1==n2
        if pre_tests:
            xdiff = x1-x2
            tn,pn = scipystats.shapiro(xdiff)
            test_results['shapiro'] = {'p-value':pn, 'stats':tn}

            if print_round>0:
                tn,pn = np.around(tn,print_round),np.around(pn,print_round)

            print(bcolors.BOLD+f'Test for Normality'+bcolors.ENDC)
            print('-'*50)
            print('Shapiro-test on diff')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'p-value: {pn} -  with stats {tn} \t|  {sss}')
            if notes: print('If test is not significant (p<alpha) that indicates the sampling distribution is normally distributed.')
            if plots:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                normality_plot, stat = scipystats.probplot(xdiff, plot= plt, rvalue= True)
                ax.set_title("Probability plot of sampling difference")
                plt.show()

        tv1,p1 = scipystats.ttest_rel(x1,x2)
        tv2,p2 = scipystats.wilcoxon(x1,x2)

        test_results['t-test'] = {'p-value':p1, 'stats':tv1}
        test_results['wilcox'] = {'p-value':p2, 'stats':tv2}

    else:
        if pre_tests:
            tn1,pn1 = scipystats.shapiro(x1)
            tn2,pn2 = scipystats.shapiro(x2)

            test_results['shapiro_x1'] = {'p-value':pn1, 'stats':tn1}
            test_results['shapiro_x2'] = {'p-value':pn2, 'stats':tn2}

            if print_round>0:
                tn1,pn1 = np.around(tn1,print_round),np.around(pn1,print_round)
                tn2,pn2 = np.around(tn2,print_round),np.around(pn2,print_round)
            print(bcolors.BOLD+f'Test for Normality'+bcolors.ENDC)
            print('-'*50)
            print('Shapiro-test')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn1>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'x1: p-value {pn1}: with stats {tn1} | {sss}')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn2>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'x2: p-value {pn2}: with stats {tn2} | {sss}')
            if notes: print('If test is not significant (p<alpha) that indicates the sampling distribution is normally distributed.')
            if plots:
                fig = plt.figure(figsize=(figsize[0]*2,figsize[1]))
                ax = fig.add_subplot(121)
                normality_plot, stat = scipystats.probplot(x1, plot= plt, rvalue= True)
                ax.set_title("Probability plot of sampling difference")
                ax = fig.add_subplot(122)
                normality_plot, stat = scipystats.probplot(x2, plot= plt, rvalue= True)
                ax.set_title("Probability plot of sampling difference")
                plt.show()

            print('')
            print(bcolors.BOLD+f'Test for Homogeneity of Variance'+bcolors.ENDC)
            print('-'*50)
            tn3, pn3 = scipystats.levene(x1,x2)

            test_results['levene'] = {'p-value':pn3, 'stats':tn3}

            if print_round>0:
                tn3,pn3 = np.around(tn3,print_round),np.around(pn3,print_round)

            print('Levene test')
            sss = bcolors.BGreen+'Pass'+bcolors.ENDC if pn3>alpha else bcolors.BRed+'Fail'+bcolors.ENDC
            print(f'p-value {pn3}: with stats {tn3} | {sss}')
            if notes: print('The small p-value suggests that the populations do not have equal variances')


        tv1,p1 = scipystats.ttest_ind(x1,x2)
        tv2,p2 = scipystats.ranksums(x1,x2)

        test_results['t-test'] = {'p-value':p1, 'stats':tv1}
        test_results['ranksum'] = {'p-value':p2, 'stats':tv2}

    if print_round>0:
        tv1,p1 = np.around(tv1,print_round),np.around(p1,print_round)
        tv2,p2 = np.around(tv2,print_round),np.around(p2,print_round)


    pprint=1

    if printthr<1:
        pprint = (p1<printthr) or (p2<printthr)

    if pprint:
        print('')
        if effect_size:
            mead_df  = x1.mean() - x2.mean()
            sd_poold = (np.var(x1,ddof=1)*(n1-1) + np.var(x2,ddof=1)*(n2-1))/(n1+n2-2)
            sd_poold = np.sqrt(sd_poold)
            cohensD  = mead_df/sd_poold

            effect_results['effect_size'] = {'mean_diff':mead_df, 'CohensD':cohensD}

            print(bcolors.BOLD+'Effect Size'+'\t'+bcolors.ENDC)
            print('-'*50)
            if print_round>0:
                print(f'Mean diff (x1-x2): \t {mead_df.round(print_round)}')
                print(f'CohensD   (x1-x2): \t {cohensD.round(print_round)}')
            else:
                print(f'Mean diff (x1-x2): \t {mead_df}')
                print(f'CohensD   (x1-x2): \t {cohensD}')
            #print('-'*50)
        print('\n')
        print('='*50)
        if title is not None:
            print(bcolors.BOLD+'Final Test : '+title+'\t'+bcolors.ENDC)
        else:
            print(bcolors.BOLD+'Final Test'+'\t'+bcolors.ENDC)
        print('='*50)

        ssts  = '\t\t(stats)' if tval else ''
        if paired:
            print(bcolors.BOLD+f'Test                 \tp-value{ssts} |(n={n})'+bcolors.ENDC)
        else:
            print(bcolors.BOLD+f'Test                 \tp-value{ssts} |(n1={n1}, n2={n2})'+bcolors.ENDC)
        print('-'*50)

        ss = _getSS(p1,alpha)

        if tval:
            if paired:
                print(ss[0]+f'T-test paired        \t{p1}\t\t(stats= {tv1})'+ss[1])
            else:
                print(ss[0]+f'T-test indept.       \t{p1}\t\t(stats= {tv1})'+ss[1])
        else:
            if paired:
                print(ss[0]+f'T-test paired        \t{p1}'+ss[1])
            else:
                print(ss[0]+f'T-test indept.       \t{p1}'+ss[1])

        ss = _getSS(p2,alpha)

        if tval:
            if paired:
                print(ss[0]+f'Wilcoxon signed-rank \t{p2}\t\t(stats= {tv2})'+ss[1])
            else:
                print(ss[0]+f'Wilcoxon rank-sum    \t{p2}\t\t(stats= {tv2})'+ss[1])
        else:
            if paired:
                print(ss[0]+f'Wilcoxon signed-rank \t{p2}'+ss[1])
            else:
                print(ss[0]+f'Wilcoxon rank-sum    \t{p2}'+ss[1])
        #print('-'*50)
        print('='*50)

        # Effect Size
        #mean diff
        #Cohen's d
        #correlation

        print('')

    df_tests = pd.DataFrame(test_results).T
    df_esize = pd.DataFrame(effect_results).T

    tPass = p1<alpha or p2<alpha
    if return_all:
        return tPass,(df_tests,df_esize)
    return tPass

def test_groups(x,axis=0,alpha=0.05,title=None,printthr=1,notes=False,print_round=4,return_all=False):
    r"""Test multiple groups: One-way Anova
    
    
    Parameters
    ----------
    x: list of groups/sample or 2d-array
      - if list x = [x1,x2,x3], for testing x1, x2, and x3 as three groups
      - if np.array, `axis`  determine the groups/sample
    
    axis: 0 or 1,
      - only used if x is np.array, 
      - to determine the sample axis

    alpha: scalar [0,1], default=0.05
      - alpha level,
      - threshold on p-value for passing/failing declaration
    
    print_round: int, default=4
      - rounding off all the numbers to decimal points
      - print_round=4 means upto 4 decimal points
      - print_round=-1 means all the decimal points available

    title: str, default=None
      - if passed as str, used as heading with "Final Test"
      - useful when running many tests
    
    printthr: scalar [0,1], deafult=1
      - threhold on p-value to display the results of final test
      - if p-value of final test is >printthr then 'final test' results are not printed
      - default=1 to always print the results of final test
    
    notes: bool, default=True,
       - if True, printing explaination

    return_all: bool, default=False
      - if True, two tables of all the results are returned

    Returns
    -------
    tPass: bool
       - True, if any one of the final test was passed (i.e., p-value < alpha)
       - False means, none of the final test was passed

    df_tests: pd.DataFrames
       - df_tests: Table of the test
    

    References
    ----------


    See Also
    --------

    Examples
    --------
    >>> #sp.stats.test_groups
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> x1 = np.random.randn(100)
    >>> x2 = np.random.randn(100)+0.15
    >>> x3 = np.random.rand(100)
    >>> tPass,df = sp.stats.test_groups(x=[x1,x2,x3],return_all=True,print_round=4,title='test-example', notes=True)
    >>> print(df)
                    p-value     stats
    One-way ANOVA  0.000111  9.388809
    """

    xi = x
    if isinstance(x,np.ndarray):
        m,n = x.shape
        xi = x.tolist() if axis==0 else x.T.tolist()
        #print(m,n)

    m = len(xi)
    n = [len(xii) for xii in xi]
    if len(set(n))==1: n = n[0]
    #print(m,n)
    ft, pv = f_oneway(*xi)
    test_results = {}
    test_results['One-way ANOVA'] = {'p-value':pv, 'stats':ft}

    if print_round>0: ft,pv = np.around(ft,print_round),np.around(pv,print_round)
    pprint=1
    if printthr<1:
        pprint = (pv<printthr)

    if pprint:
        if title is not None:
            print(bcolors.BOLD+title+'\t'+bcolors.ENDC)
            print('='*25)
        
        space1 = " "*(13-4)
        space2 = " "*(len(str(ft))+2-5)
        print(bcolors.BOLD+f'Test{space1} \tstats{space1}  \tp-value (m={m}, n={n})'+bcolors.ENDC)
        print('-'*70)
        ss = _getSS(pv,alpha)
        print(ss[0]+f'One-way ANOVA \tF:{ft} \t{pv}'+ss[1])
        print('-'*70)
        if notes:
            str_note = f'Given {m} groups with number of samples in each groups {n}, test shows p-value = {pv}.\n'
            str_note1 = f'Which suggestes that there are at least two groups which are significantly different from each other. \nRun post-hoc analysis to identify those pairs of groups.'
            str_note2 = f'Which suggestes that all the {m} groups are similar to each other.'

            if pv<alpha:
                str_note = str_note +str_note1
            else:
                str_note = str_note +str_note2

            print('NOTES:'+str_note)

    tPass= pv<alpha
    df_test = pd.DataFrame(test_results).T
    if return_all:
        return tPass, df_test
    return tPass

def plotBoxes_groups(xi,lines=False,xlabels=[],ylab='',title='',dodge=True,notch=False,showmeans=True,ax=None,jitter=0,
                 showBox=True,lines_groups=[0,1],group_c=['k','C1','C2'],box_palette=None,
                 ms_color='k',ms_size=3,
                 line_keyworks={'lw':0.5,'ls':'--'},
                 meanprops={'ms':10,'marker':'o','markerfacecolor':'r'},**boxprop):

    if lines:
        if isinstance(xi,list):
            if len(set([len(xii) for xii in xi]))==1:
                xi = np.array(xi).T
            else:
                lines=False
                print('Lines can not be drawn for un-paired samples, i.e. samples in each group should be same')

    if ax is None: fig, ax = plt.subplots()
    if showBox:
        ax = sns.boxplot(data=xi,width=0.5,dodge=dodge,palette=box_palette,
                     notch=notch,showmeans=showmeans,meanprops=meanprops,ax=ax,**boxprop)
    if lines:
        for i in range(xi.shape[1]):
            ax.plot(xi[:,0]*0+i,xi[:,i],'.'+ms_color,ms=ms_size)

        if len(lines_groups):
            idx = xi[:,lines_groups[0]]>xi[:,lines_groups[1]]
            if sum(idx): ax.plot(xi[idx].T,color=group_c[0],**line_keyworks)
            if sum(~idx): ax.plot(xi[~idx].T,color=group_c[1],**line_keyworks)
        else:
            ax.plot(xi.T,color=group_c[0],**line_keyworks)
    else:
        ax = sns.stripplot(data=xi,color="0.1")
    plt.grid()
    if isinstance(xi, list):
        plt.xticks(range(len(xi)),xlabels)
    else:
        plt.xticks(range(xi.shape[1]),xlabels)
    plt.ylabel(ylab)
    plt.title(title)
    return ax

def plot_groups_boxes(x,lines=False,xlabels=[],ylab='',title='',ax=None,
                      box_palette=None,dodge='auto',notch=False,
                      showmeans=True,show_box=True,
                      lines_groups=[],group_colors=[],
                      **kwargs):
    r"""Boxplot for Groups data

    Parameters
    ----------
    x: list of arrays
       - data groups
    lines: bool, default=False
       - if True and if Data is paired (number of samples in each group are same), 
          lines are drawn to show the trends
    
    xlabels: list of str
       -  names for each group, 
       - should have same length as number of groups
    
    ylab: str, default=''
      - label for ylabel
    
    title: str, default=''
      - tilte of figure

    ax: ax obj, default=None
      -  None then created using `fig, ax = plt.subplots()`

    box_palette: str, list of str, default=None
      - color palette for boxes
      - if None, default color palette
      - example: box_palette on of `{'pastel','husl','Set2','Spectral','flare'}`
      - Or - `box_palette = ['C0','C1', 'C2' ..]`
    
    dodge: bool,default='auto',
      - to avoid overlap, 'auto' sets itself
    
    notch: bool, default=False,
      - if False, then rectangular boxes as boxplot
      - if True triangular shape style
    
    showmeans: bool, default=True,
      - If True, show mean of the data, with properties `meanprops` 
    
    show_box: bool, default=True,
      - if False, boxes are not show, 
      - useful to show points only, 
                      
    lines_groups: list, default=[]
      - grouping the lines with colours
      - for example:
           - lines_groups = [0,1] means points where 
           - group 1 is greater 2, it is one group
           - see example
    
    group_colors: list, default=[]
      -  colours for groups of lines

    kwargs:
    
        There are other arguments which can be supplied to modify the plots.
        
        Default setting is:
        * `line_kw=dict(lw=0.5,ls='--',color='k')`  for line
        * `line_marker_kw = dict(marker='.',ms=3,color='k',lw=0)` for dot on lines
        * `meanprops=dict(ms=10,marker='o',markerfacecolor='r')` for means
        * `box_kw =dict(width=0.5)` for boxes
        * `strip_kw=dict(color="0.1")` for dots when lines are not used

    Return
    ------
    ax: matplotlib obj
      - return the axis to use for the modify
    

    References
    ----------
    * .. [1] https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette


    Examples
    --------
    #sp.stats.plot_groups_boxes
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    np.random.seed(1)
    x1 = np.random.randn(50)
    x2 = np.random.randn(40)+1
    x3 = np.random.rand(40) +1 
    x4 = np.random.rand(40)

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sp.stats.plot_groups_boxes(x= [x1,x2], ax=ax[0],title='2 Unpaired groups',ylab='Values')
    sp.stats.plot_groups_boxes(x= [x2,x3,x4],ax=ax[1], lines=True, xlabels=['A','B','C'],title='3 paired groups')
    plt.show()
    """

    props = dict(line_kw=dict(lw=0.5,ls='--',color='k'),
                 line_marker_kw = dict(marker='.',ms=3,color='k',lw=0),
                 meanprops=dict(ms=10,marker='o',markerfacecolor='r'),
                 box_kw= dict(width=0.5),
                 strip_kw=dict(color="0.1"))
    
    
    # Updating Settings
    if 'meanprops' in kwargs:
        for key in props['meanprops']:
            if key not in kwargs['meanprops']:
                kwargs['meanprops'][key] = props['meanprops'][key]

    for key in kwargs:
        props[key] = kwargs[key]
        
    try:
        import seaborn as sns
    except:
        raise ImportError("This function required seaborn library, install it using 'pip install seaborn' ")
    
    if lines:
        if isinstance(x,list):
            if len(set([len(xii) for xii in x]))==1:
                x = np.array(x).T
                #print(x.shape)
            else:
                lines=False
                warnings.warn('Lines can not be drawn for un-paired samples, i.e. samples in each group should be same')

    if ax is None: fig, ax = plt.subplots()
    if show_box:
        ax = sns.boxplot(data=x,dodge=dodge,palette=box_palette,ax=ax,
                     notch=notch,showmeans=showmeans,meanprops=props['meanprops'],**props['box_kw'])
    if lines:
        line_kw = copy.deepcopy(props['line_kw'])
        
        for i in range(x.shape[1]):
            ax.plot(x[:,0]*0+i,x[:,i],**props['line_marker_kw'])

        if len(lines_groups)==2:
            if not(len(group_colors)==2):
                group_colors = ['C1','C2']
            
            line_kw_g = copy.deepcopy(props['line_kw'])
            
            if 'color' in line_kw_g:
                del line_kw_g['color']
                
                
            idx = x[:,lines_groups[0]]>x[:,lines_groups[1]]
            
            if sum(idx): ax.plot(x[idx].T,color=group_colors[0],**line_kw_g)
            
            if sum(~idx): ax.plot(x[~idx].T,color=group_colors[1],**line_kw_g)
        else:
            ax.plot(x.T,**line_kw)
    else:
        ax = sns.stripplot(data=x,ax=ax,**props['strip_kw'])

    nG  = len(x) if isinstance(x, list) else x.shape[1]
    
    #if isinstance(x, list):
    #    nG  = len(x)
    #else:
    #    nG  = x.shape[1]

    if len(xlabels)==0:
        xlabels = np.arange(nG).astype(int)
    
    ax.set_xticks(np.arange(nG).astype(int),xlabels)

    ax.grid()
    ax.set_ylabel(ylab)
    ax.set_title(title)
    return ax


# -------