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
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy import stats as scipystats


from ..utils import bcolors
from ..core.infotheory import entropy

# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#     CRED = '\033[91m'
#     BGreen = '\x1b[6;10;42m'
#     BGrey  = '\x1b[6;10;47m'
#     BRed = '\x1b[6;10;41m'
#     BYellow = '\x1b[6;10;43m'
#     BEND = '\x1b[0m'

def quick_stats(x,show=True,return_table=False,name='x'):
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

def get_stats(x,detail_level=1,return_names=False,name='x'):
    '''
    Statistics of a given sequence x, excluding NaN values
    ------------------------------------------------------
    returns stats and names of statistics measures

    '''
    stats_names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    #names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    esp=1e-5
    if isinstance(x,int) or isinstance(x,float): x =  [x]
    if isinstance(x,list):x = np.array(x)
    assert len(x.shape)==1
    #logsum = self.get_exp_log_sum(x)

    x = x+esp
    mn = np.nanmean(x)
    sd = np.nanstd(x)
    md = np.nanmedian(x)
    min0 = np.nanmin(x)
    max0 = np.nanmax(x)

    n = len(x) - sum(np.isnan(x))

    if detail_level==1:
        S = np.r_[mn,sd,md,min0,max0,n]
        Sdf = pd.DataFrame(S,index=stats_names[:6],columns=[name]).T
        return S, stats_names[:6], Sdf

    q25 = np.nanquantile(x,0.25)
    q75 = np.nanquantile(x,0.75)
    iqr = scipystats.iqr(x[~np.isnan(x)])
    kur = scipystats.kurtosis(x,nan_policy='omit')
    skw = scipystats.skew(x[~np.isnan(x)])
    if detail_level==2:
        S =np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw]
        Sdf = pd.DataFrame(S,index=stats_names[:11],columns=[name]).T
        return S, stats_names[:11], Sdf

    gmn = scipystats.gmean(x[~np.isnan(x)])
    entropy = entropy(x[~np.isnan(x)])

    S = np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw,gmn,entropy]
    Sdf = pd.DataFrame(S,index=stats_names,columns=[name]).T
    return S, stats_names, Sdf

def outliers(x, method='iqr',k=1.5, include_lower=True, include_upper=True,return_thr = False):
    #Atleast one of limit should be true
    assert (include_upper+include_lower)
    xi = x.copy()
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
    if p<=thr:
        ss = (bcolors.BGreen,bcolors.BEND)
    elif p<=0.1:
        ss = (bcolors.BGrey,bcolors.BEND)
    return ss

def test_2groups(x1,x2,paired=True,alpha=0.05,title=None,tval=False,printthr=1,return_all=False,
                    print_round=-1,notes=True,pre_tests=True,effect_size=True,plots=True):

    r"""

    Test two groups
    ----------------

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
            print('-------------------')
            print('Shapiro-test on diff')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'p-value: {pn} -  with stats {tn} \t|  {sss}')
            if notes: print('If test is not significant (p<alpha) that indicates the sampling distribution is normally distributed.')
            if plots:
                fig = plt.figure()
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
            print('-------------------')
            print('Shapiro-test')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn1>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'x1: p-value {pn1}: with stats {tn1} | {sss}')
            sss = bcolors.BGreen+'  Pass  '+bcolors.ENDC if pn2>alpha else bcolors.BRed+'  Fail  '+bcolors.ENDC
            print(f'x2: p-value {pn2}: with stats {tn2} | {sss}')
            if notes: print('If test is not significant (p<alpha) that indicates the sampling distribution is normally distributed.')
            if plots:
                fig = plt.figure(figsize=(10,4))
                ax = fig.add_subplot(121)
                normality_plot, stat = scipystats.probplot(x1, plot= plt, rvalue= True)
                ax.set_title("Probability plot of sampling difference")
                ax = fig.add_subplot(122)
                normality_plot, stat = scipystats.probplot(x2, plot= plt, rvalue= True)
                ax.set_title("Probability plot of sampling difference")
                plt.show()

            print('')
            print(bcolors.BOLD+f'Test for Homogeneity of Variance'+bcolors.ENDC)
            print('-------------------')
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
            ##print('-'*50)
        print('\n'*2)
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

    df1 = pd.DataFrame(test_results).T
    df2 = pd.DataFrame(effect_results).T

    tPass = p1<=alpha or p2<=alpha
    if return_all:
        return tPass,df1,df2
    return tPass

def test_groups(x,axis=0,alpha=0.05, title=None,printthr=1,return_all = False,print_round=-1):
    r"""
    Test multiple groups: One-way Anova
    ---------------------------------


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
        print(bcolors.BOLD+f'Test        \t\tstats            \tp-value (m={m}, n={n})'+bcolors.ENDC)
        print('-'*70)
        ss = _getSS(pv,alpha)
        print(ss[0]+f'One-way ANOVA     \tF:{ft}   \t{pv}'+ss[1])
        print('-'*70)

    tPass= pv<alpha
    if return_all:
        return tPass, pd.DataFram(test_results).T
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
