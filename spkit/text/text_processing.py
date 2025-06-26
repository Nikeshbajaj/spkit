import re
import numpy as np

def clean_html(line):
    r"""Clean html line and extract text

    remove everything that is between < and >, including them
    as majority of the html code formating is writting < >, using this simple approach can remove all the code, 
    and leave raw text.
    """

    pattern1 = "<"
    pattern2 = ">"
    idx1 = np.array([m.start(0) for m in re.finditer(pattern1, line)])
    idx2 = np.array([m.start(0) for m in re.finditer(pattern2, line)])
    idx2 = idx2[idx2>idx1[0]] # making sure, it starts with <
    minlen = min(len(idx1),len(idx2))
    
    lk = ''
    if idx1[0]>0:
        lk = lk + line[:idx1[0]]
    for i in range(minlen-1):
        li = line[idx2[i]+1:idx1[i+1]]
        if len(li): lk = lk + li
    return lk.strip()