from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | utils"
import sys, os

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
from scipy  import stats as scipystats

# RELATIVE PATH NAMED TO BE CHANGED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# sys.path.append("..")
# from spkit.core.information_theory import entropy
# from spkit.utils_misc.borrowed import resize
# #
#sys.path.append(os.path.dirname('.'))
#from .core.information_theory import entropy
#from .utils_misc.borrowed import resize

# sys.path.append(os.path.dirname(__file__))
# from .core.information_theory import entropy
# from .utils_misc.borrowed import resize

#sys.path.append("..")
#from .utils_misc.borrowed import resize

import functools, inspect, warnings


class txcolors:
    r"""Utilities for coloring text

    Try these examples with Jupyter-Notebook or on Terminal.

    See Also
    --------
    bcolors

    Examples
    --------
    >>> import spkit as sp
    >>> TXT = sp.utils.txcolors
    >>> print(TXT.CRED + 'Red Font' +TXT.ENDC)
    >>> print(TXT.BYellow + 'Yellow Background' +TXT.ENDC)
    >>> print(TXT.CBLUE + 'Blue Font' +TXT.ENDC)
    >>> print(TXT.OKGREEN + 'Blue Font' +TXT.ENDC)
    """
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
    #--new------------
    CBLACK  = '\33[30m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'
    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'
    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

class bcolors:
    r"""Utilities for coloring text

    Try these examples with Jupyter-Notebook or on Terminal.

    See Also
    --------
    txcolors

    Examples
    --------
    >>> import spkit as sp
    >>> TXT = sp.utils.bcolors
    >>> print(TXT.CRED + 'Red Font' +TXT.ENDC)
    >>> print(TXT.BYellow + 'Yellow Background' +TXT.ENDC)
    >>> print(TXT.CBLUE + 'Blue Font' +TXT.ENDC)
    >>> print(TXT.OKGREEN + 'Blue Font' +TXT.ENDC)
    """
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
    #--new------------
    CBLACK  = '\33[30m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'
    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'
    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

A=['\\','-','/','|']

def ProgBar_JL(i,N,title='',style=2,L=50,selfTerminate=True,delta=None,sym='▓',color='blue'):
    r"""Progress Bar : Utilities - Compatible with JupyterLab/Jupyter-Notebook
    
    Progress Bar
    
    ▇ ▓ ▒ ░ ▉

    Parameters
    ----------
    i: int, float
      - current index
    N: int, float,
      - Final number
    
    sym: symbol, default='▓'
      - symbol as bar {▇ ▓ ▒ ░ ▉}
    
    color: str,
      - one of {'green','blue','cyan','red','yellow'}

    title: str,
      -  title to show end of the bar

    style: int, {1,2}, default=2
      - different styles

    L: int {100,50}
      - length of Bar

    selfTerminate: bool, default=True
      - if i>N, it terminates

    Returns
    -------
    display :  bars

    See Also
    --------
    ProgBar

    Examples
    --------
    >>> #sp.utils.ProgBar_JL
    >>> import time
    >>> import numpy as np
    >>> import spkit as sp
    >>> N = 340 
    >>> for i in range(N):
    >>>     sp.utils.ProgBar_JL(i,N, style=2,sym='▓',color='blue',title='example-1')
    100%|▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓|340\340| example-1
    Done!
    """
    c1 = bcolors.ENDC
    if color.lower() in ['green','blue','cyan','red','yellow']:
        if color.lower()=='green':
            c1 = bcolors.OKGREEN
        elif color.lower()=='blue':
            c1 = bcolors.OKBLUE
        elif color.lower()=='cyan':
            c1 = bcolors.OKCYAN
        elif color.lower()=='red':
            c1 = bcolors.CRED
        elif color.lower()=='yellow':
            c1 = bcolors.CYELLOW

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
        print('\nDone!')

def ProgBar(i,N,title='',style=2,L=50,selfTerminate=False,sym='▓',color='blue'):
    r"""Progress Bar : Utilities
    
    Progress Bar
    
    ▇ ▓ ▒ ░ ▉

    Parameters
    ----------
    i: int, float
      - current index
    N: int, float,
      - Final number
    
    sym: symbol, default='▓'
      - symbol as bar {▇ ▓ ▒ ░ ▉}
    
    color: str,
      - one of {'green','blue','cyan','red','yellow'}

    title: str,
      -  title to show end of the bar

    style: int, {1,2}, default=2
      - different styles

    L: int {100,50}
      - length of Bar

    selfTerminate: bool, default=True
      - if i>N, it terminates

    Returns
    -------
    display :  bars

    See Also
    --------
    ProgBar

    Examples
    --------
    >>> #sp.utils.ProgBar
    >>> import time
    >>> import numpy as np
    >>> import spkit as sp
    >>> N = 340 
    >>> for i in range(N):
    >>>     sp.utils.ProgBar(i,N, style=2,sym='▓',color='blue',title='example-1')
    100%|▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓|340\340| example-1
    Done!
    """
    c1 = bcolors.ENDC
    if color.lower() in ['green','blue','cyan','red','yellow']:
        if color.lower()=='green':
            c1 = bcolors.OKGREEN
        elif color.lower()=='blue':
            c1 = bcolors.OKBLUE
        elif color.lower()=='cyan':
            c1 = bcolors.OKCYAN
        elif color.lower()=='red':
            c1 = bcolors.CRED
        elif color.lower()=='yellow':
            c1 = bcolors.CYELLOW

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
        print('\nDone!')

def pretty_print(List,n=3,sep='|\t',show_index=True,trimLength=None):
    r"""Print list : in pretty way 
    
    Displaying list with index and in columns-wise

    Parameters
    ----------
    List: list of str
      - list of str

    n: int, default=3
      - number of columns

    sep: default='|\t'
      - seperator of the columns

    show_index: bool, default=True
      - show index or not
    
    trimLength: int, default=None,
      - triming the str of each item in list to look good

    Returns
    -------
    display: show list

    Examples
    --------
    >>> #sp.utils.pretty_print
    >>> import time
    >>> import numpy as np
    >>> import spkit as sp
    >>> Ch_Names = list(sp.eeg.presets.standard_1020_ch)[:50]
    >>> sp.utils.pretty_print(Ch_Names, n=5)
        0 LPA  |	1 RPA  |	2 Nz   |	3 Fp1  |	4 Fpz  
        5 Fp2  |	6 AF9  |	7 AF7  |	8 AF5  |	9 AF3  
        10 AF1 |	11 AFz |	12 AF2 |	13 AF4 |	14 AF6 
        15 AF8 |	16 AF10|	17 F9  |	18 F7  |	19 F5  
        20 F3  |	21 F1  |	22 Fz  |	23 F2  |	24 F4  
        25 F6  |	26 F8  |	27 F10 |	28 FT9 |	29 FT7 
        30 FC5 |	31 FC3 |	32 FC1 |	33 FCz |	34 FC2 
        35 FC4 |	36 FC6 |	37 FT8 |	38 FT10|	39 T9  
        40 T7  |	41 C5  |	42 C3  |	43 C1  |	44 Cz  
        45 C2  |	46 C4  |	47 C6  |	48 T8  |	49 T10 
    >>> sp.utils.pretty_print(Ch_Names, n=3)
        LPA |	RPA |	Nz  
        Fp1 |	Fpz |	Fp2 
        AF9 |	AF7 |	AF5 
        AF3 |	AF1 |	AFz 
        AF2 |	AF4 |	AF6 
        AF8 |	AF10|	F9  
        F7  |	F5  |	F3  
        F1  |	Fz  |	F2  
        F4  |	F6  |	F8  
        F10 |	FT9 |	FT7 
        FC5 |	FC3 |	FC1 
        FCz |	FC2 |	FC4 
        FC6 |	FT8 |	FT10
        T9  |	T7  |	C5  
        C3  |	C1  |	Cz  
        C2  |	C4  |	C6  
        T8  |	T10 
    """
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

def view_hierarchical_order(file_obj,sep=' ├── ',level=0,st='',show=True,return_str=False):
    r"""View nested dictionary type objects as hierarchical tree-order
    
    View nested dictionary type objects as hierarchical tree-order
    
    Parameters
    ----------
    file_obj: dict-like
      - dictionary type object, object which has keys as attribuete

    sep: str, default = '-->|'
      - seperation style
      - {'-->|', ' ├──' }

    show:bool, 
      - if false, then tree is not printed

    return_str: bool, default=False
      - if true, return a printable-formated-string to reproduce tree order


    Returns
    -------
    st : str
        Tree-order as string, can be used to reproduce tree by print(st).
        if return_str is true

    Examples
    --------
    >>> #sp.utils.view_hierarchical_order
    >>> import spkit as sp
    >>> dObj = {'class 1':1, 'class 2':{'class 2.1':21, 
    >>>        'class 2.2':{'class 2.2.1':4}},'class 3':3,
    >>>        'class 4':{'class 4.1':41,}}
    >>> sp.utils.view_hierarchical_order(dObj, sep='-->|')
    |class 1
    |class 2
    |-->|class 2.1
    |-->|class 2.2
    |-->|-->|class 2.2.1
    |class 3
    |class 4
    |-->|class 4.1
    >>> sp.utils.view_hierarchical_order(dObj, sep=' ├──')
    |class 1
    |class 2
    | ├──class 2.1
    | ├──class 2.2
    | ├── ├──class 2.2.1
    |class 3
    |class 4
    | ├──class 4.1
    """
    if hasattr(file_obj, 'keys'):
        for key in file_obj.keys():
            st = st+f'|{sep*level}{key}\n'
            if show: print(f'|{sep*level}{key}')
            st = view_hierarchical_order(file_obj[key],sep=sep,level=level+1,st=st,show=show,return_str=True)

    if return_str: return st

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line

string_types = (type(b''), type(u''))

def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "class {name} will be deprecated in future version, {reason}."
            else:
                fmt1 = "function {name} will be deprecated in future version, {reason}."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "class {name} will be deprecated in future version."
        else:
            fmt2 = "function {name} will be deprecated in future version."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))

#----------Under Dev------------------
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
        print('\nDone!')

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
        print('\nDone!')

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
        print('\nDone!')

def print_list(L,n=3,sep='\t\t'):
    """Print list :  Utilities
    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia
    
    
    Notes
    -----

    See Also
    --------
    spkit: # TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    """
    L = [str(l) for l in L]
    mlen = np.max([len(ll) for ll in L])
    for k in range(0,len(L)-n,n):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,k+n)]))
    if k+n<len(L):
        print(sep.join([L[ki] +' '*(mlen-len(L[ki])) for ki in range(k,len(L))]))
