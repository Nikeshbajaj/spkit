from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | utils"
import sys

if sys.version_info[:2] < (3, 3):
    old_print = print
    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        if flush:
            file = kwargs.get('file', sys.stdout)
            # Why might file=None? IDK, but it works for print(i, file=None)
            file.flush() if file is not None else sys.stdout.flush()

import numpy as np
from scipy  import stats
from ..core.information_theory import entropy
import warnings

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    CRED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BGreen = '\x1b[6;10;42m'
    BGrey  = '\x1b[6;10;47m'
    BRed = '\x1b[6;10;41m'
    BYellow = '\x1b[6;10;43m'
    BEND = '\x1b[0m'

A=['\\','-','/','|']

def ProgBar_JL(i,N,title='',style=2,L=50,selfTerminate=True,delta=None,sym='▓',color='green'):
    '''
    ▇ ▓ ▒ ░ ▉
    '''
    c1 = bcolors.ENDC
    if color.lower() in ['green','blue','cyan','red']:
        if color.lower()=='green':
            c1 = bcolors.OKGREEN
        elif color.lower()=='blue':
            c1 = bcolors.OKBLUE
        elif color.lower()=='cyan':
            c1 = bcolors.OKCYAN
        elif color.lower()=='red':
            c1 = bcolors.CRED

    c2 = bcolors.ENDC


    pf = int(100*(i+1)/float(N))
    st = '\r'+' '*(3-len(str(pf))) + str(pf) +'%|'

    if L==50:
        pb = sym*int(pf//2)+' '*(L-int(pf//2))
    else:
        L = 100
        pb = sym*pf+' '*(L-pf)

    pb = c1 + pb + c2 +'|'

    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='', flush=True)
    if pf>=100 and selfTerminate:
        print('\nDone..')

def ProgBar(i,N,title='',style=2,L=50,selfTerminate=False,sym='▓',color='green'):
    '''
    ▇ ▓ ▒ ░ ▉
    '''
    c1 = bcolors.ENDC
    if color.lower() in ['green','blue','cyan','red']:
        if color.lower()=='green':
            c1 = bcolors.OKGREEN
        elif color.lower()=='blue':
            c1 = bcolors.OKBLUE
        elif color.lower()=='cyan':
            c1 = bcolors.OKCYAN
        elif color.lower()=='red':
            c1 = bcolors.CRED

    c2 = bcolors.ENDC

    pf = int(100*(i+1)/float(N))
    st = ' '*(3-len(str(pf))) + str(pf) +'%|'

    if L==50:
        pb = sym*int(pf//2)+' '*(L-int(pf//2))
    else:
        L = 100
        pb = sym*pf+' '*(L-pf)
    pb = c1 + pb + c2 +'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='\r', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='\r', flush=True)
    if pf>=100 and selfTerminate:
        print('\nDone..')

def ProgBar_JL_v0(i,N,title='',style=2,L=100,selfTerminate=True,delta=None):
    pf = int(100*(i+1)/float(N))
    st = '\r'+' '*(3-len(str(pf))) + str(pf) +'%|'

    if L==50:
        pb = '#'*int(pf//2)+' '*(L-int(pf//2))+'|'
    else:
        L = 100
        pb = '#'*pf+' '*(L-pf)+'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='', flush=True)
    if pf>=100 and selfTerminate:
        print('\nDone..')

def ProgStatus(i,N, title='',style=1,speed=5):
    pt = int(100*(i+1)/N)
    A = ['\\','|','/','-']
    D = ['.\t','..\t','...\t','....\t','....\t']

    st0 = ' '*10
    if style==1:
        st = 'Computating...'+A[(pt//speed)%len(A)]+' '+str(pt)+'%\t| '
    else:
        #st = 'Computating'+D[i%len(D)]+' '+str(pt)+'%\t| '
        st = 'Computating'+D[(pt//speed)%len(D)]+' '+str(pt)+'%\t| '

    print(st+title+st0,end='\r',flush=True)

def ProgBar_v0(i,N,title='',style=2,L=100,selfTerminate=True,delta=None):

    pf = int(100*(i+1)/float(N))
    st = ' '*(3-len(str(pf))) + str(pf) +'%|'

    if L==50:
        pb = '#'*int(pf//2)+' '*(L-int(pf//2))+'|'
    else:
        L = 100
        pb = '#'*pf+' '*(L-pf)+'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='\r', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='\r', flush=True)
    if pf>=100 and selfTerminate:
        print('\nDone..')

def ProgBar_float(i,N,title='',style=2,L=100,selfTerminate=True,delta=None):
    pf = np.around(100*(i+1)/float(N),2)
    st = ' '*(5-len(str(pf))) + str(pf) +'%|'
    if L==50:
        pb = '#'*int(pf//2)+' '*(L-int(pf//2))+'|'
    else:
        L = 100
        pb = '#'*int(pf)+' '*(L-int(pf))+'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='\r', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='\r', flush=True)
    if pf>=100 and selfTerminate:
        print('\nDone..')

def print_list(L,n=3,sep='\t\t'):
    L = [str(l) for l in L]
    mlen = np.max([len(ll) for ll in L])
    for k in range(0,len(L)-n,n):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,k+n)]))
    if k+n<len(L):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,len(L))]))

def pretty_print(List,n=3,sep='|\t',show_index=True,trimLength=None):
    List = [str(l) for l in List]
    for l in List: assert type(l)==str
    L = List.copy()
    if show_index: L = [str(i)+' '+ L[i] for i in range(len(L))]

    if trimLength is not None:
        L = [ll[:trimLength] for ll in L]
    else:
        mlen = np.max([len(ll) for ll in L])
        L = [ll+' '*(mlen-len(ll)) for ll in L]

    for k in range(0,len(L)-n,n):
        print(sep.join([L[ki] for ki in range(k,k+n)]))
    if k+n<len(L):
        print(sep.join([L[ki] for ki in range(k+n,len(L))]))

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line

def view_hierarchical_order(file_obj,sep='-->|',level=0,st='',show=True,return_str=False):
    r"""
    View nested dictionary type objects as hierarchical tree-order
    --------------------------------------------------------------

    Parameters
    ----------
    file_obj: dictionary type object, object which has keys as attribuete

    sep: seperation style, 'default -->|'

    show: if false, then tree is not printed

    return_str: if true, return a printable-formated-string to reproduce tree order

    Returns
    -------
    st : str
        Tree-order as string, can be used to reproduce tree by print(st).
        if return_str is true


    Examples
    --------
    >>> import spkit as sp
    >>> dObj = {'class 1':1, 'class 2':{'class 2.1':21, 'class 2.2':{'class 2.2.1':4}},'class 3':3, 'class 4':{'class 4.1':41,}}
    >>> view_hierarchical_order(dObj)

    """
    if hasattr(file_obj, 'keys'):
        for key in file_obj.keys():
            st = st+f'|{sep*level}{key}\n'
            if show: print(f'|{sep*level}{key}')
            st = view_hierarchical_order(file_obj[key],sep=sep,level=level+1,st=st,show=show,return_str=True)

    if return_str: return st
