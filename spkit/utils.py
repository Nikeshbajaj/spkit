from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | CWT"
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
from infotheory import entropy

A=['\\','-','/','|']

def ProgBar(i,N,title='',style=1,L=50,selfTerminate=True,delta=None):

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

def ProgBar_float(i,N,title='',style=1,L=50,selfTerminate=True,delta=None):
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
    mlen = np.max([len(ll) for ll in L])
    for k in range(0,len(L)-n,n):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,k+n)]))
    if k+n<len(L):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,len(L))]))

def pretty_print(List,n=3,sep='|\t',show_index=True,trimLength=None):
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
