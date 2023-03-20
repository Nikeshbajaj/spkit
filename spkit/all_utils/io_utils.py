r"""
BASIC IO Funtions

For reading and writing files

"""


import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy, copy #
import h5py, pyedflib
from .utils import view_hierarchical_order, warnings

# def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
#     return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
#
# warnings.formatwarning = warning_on_one_line
#
# def view_hierarchical_order(file_obj,sep='-->|',level=0,st='',show=True,return_str=False):
#     r"""
#     View nested dictionary type objects as hierarchical tree-order
#     --------------------------------------------------------------
#
#     Parameters
#     ----------
#     file_obj: dictionary type object, object which has keys as attribuete
#
#     sep: seperation style, 'default -->|'
#
#     show: if false, then tree is not printed
#
#     return_str: if true, return a printable-formated-string to reproduce tree order
#
#     Returns
#     -------
#     st : str
#         Tree-order as string, can be used to reproduce tree by print(st).
#         if return_str is true
#
#
#     Examples
#     --------
#     >>> import spkit as sp
#     >>> dObj = {'class 1':1, 'class 2':{'class 2.1':21, 'class 2.2':{'class 2.2.1':4}},'class 3':3, 'class 4':{'class 4.1':41,}}
#     >>> view_hierarchical_order(dObj)
#
#     """
#     if hasattr(file_obj, 'keys'):
#         for key in file_obj.keys():
#             st = st+f'|{sep*level}{key}\n'
#             if show: print(f'|{sep*level}{key}')
#             st = view_hierarchical_order(file_obj[key],sep=sep,level=level+1,st=st,show=show,return_str=True)
#
#     if return_str: return st


def read_hdf(file_name,fs=25000,max_volt=1000,
             base_key='Data',
             signal_hkeys=['Recording_0','AnalogStream','Stream_0'],
             signal_key = 'ChannelData',
             info_key ='InfoChannel',
             verbose=False):
    r"""

    Reading HDF File (default setting for MEA HDF files)
    ----------------------------------------------------
    Hierarchical Data Format


    Parameters
    ----------
    file_name: full path of hdf file
    fs       : (default=25KHz) sampling frequency

    max_volt : (default=1000) used while scaling the voltage of signal. It is the maximum voltage of stimulus.
                If max_volt=None, no scalling is performed and raw values are returned,
                Default datatype of signal recoridng is usually 'int32', which leades to range from -32768 to 32767
                if max_volt is not None, then return signal values will be float32 and from -max_volt to max_volt

    base_key    : First key of Hierarchical formatting, (default='Data')
    signal_hkeys: Sequence of keys, where signal of each recorded channel is stored. Exclude the final key of signals, (default=['Recording_0','AnalogStream','Stream_0'])
    signal_key  : Final key for signals, (default='ChannelData')
    info_key    : key for channel information, stored in signal_hkeys, for each channel, (default='InfoChannel')

    verbose: To display file and signal info

    Returns
    -------
    X     : array of (n_ch,n), n_ch = number of channels (i.e. 60), n number of samples, if keys to hierarchical is given correctly, else file object is returned as X
    fs    : Sampling frequency, same as passed to function
    ch_labels : labels of each channel, if correct keys are provided, else, either None, or info of each channel is returned



    Examples
    --------
    >>> from spkit.io import read_hdf
    >>> file_name = 'full_accessible_path/to_hdf_file.hdf'
    >>> X,fs,ch_labels = read_hdf(file_name)


    See Also
    --------
    read_bdf, read_surf_file, write_vtk


    """
    stacklevel=2
    f = h5py.File(file_name, 'r')
    try:
        if verbose: print('base key(s) found in file',f.keys())
        ch_data = f[base_key]
    except:
        ch_data = f
        warnings.warn('"base_key" is either missing or wrong not in Hierarchical format. \n Try "view_hierarchical_order" to see the Hierarchical order of data and use correct base_key and sequence keys',stacklevel=2)


    try:
        for hkey in signal_hkeys:
            ch_data = ch_data[hkey]
    except:
        wst = 'Some of the keys is "signal_hkeys" are either not in order or not in format\n'
        wst = wst+ 'Related to signals, check keys as in Hierarchical order for Data Format of the file \n'
        wst = wst+ 'Following order was found in the given HDF file\n'
        st = view_hierarchical_order(f,show=False)
        wst = wst+ '\n'+st+'\n Use the correct order of keys'
        warnings.warn(wst,stacklevel=stacklevel)

    try:
        X = np.array(ch_data[signal_key])
        if max_volt is not None:
            X = X/np.nanmax(X)
            X = max_volt*X
        if verbose:
            print('Shape of Signals: ',X.shape)
            print('- #Channels    =',X.shape[0])
            print('- Duration (s) =',X.shape[1]/fs)

    except:
        warnings.warn('"signal_key" is not in Hierarchical order, object is returned instead',stacklevel=stacklevel)
        X = ch_data

    try:
        info  =ch_data[info_key]
    except:
        warnings.warn('"info_key" is not in Hierarchical order',stacklevel=stacklevel)
        info = None

    ch_labels = None
    if info is not None:
        try:
            ch_labels = np.array([int(info[k][4].decode()) for k in range(len(info))])
        except:
            ch_labels = [info[k] for k in range(len(info))]
            warnings.warn('channel labels were not etracted as expected. "info_key" is corresponding to each channel is returned as ch_labels, instead',stacklevel=stacklevel)

    return X,fs,ch_labels

def read_surf_file(file_name):
    ff =  open(file_name,'rb')
    L = ff.readlines()
    for n in range(len(L)):
        if str(L[n]).count('Vertices'):
            break

    N = int(L[n].decode('ascii').split()[1])
    V = np.array([L[n+1+i].decode('ascii').split() for i in range(N)]).astype(float)

    for n in range(len(L)):
        if str(L[n]).count('Triangles'):
            break

    N = int(L[n].decode('ascii').split()[1])
    F = np.array([L[n+1+i].decode('ascii').split() for i in range(N)]).astype(float)

    return V,F

def write_vtk(filename, vertices, faces):
    """Writes .vtk file format for the Paraview (Kitware (c)) visualisation software.

    It relies on the VTK library for its writer. VTK files use the legagy ASCII file format of the VTK library.

    Parameters
    ----------
    filename: str
        name of the mesh file to be written on disk
    vertices: ndarray
        numpy array of the coordinates of the mesh's nodes
    faces: ndarray
        numpy array of the faces' nodes connectivities
    """

    nv = vertices.shape[0]
    nf = faces.shape[0]

    if faces.shape[1]==3:
        faces = np.c_[faces, faces[:,-1]]

    triangle_mask = (faces[:, 0] == faces[:, -1])
    quadrangles_mask = np.invert(triangle_mask)
    nb_triangles = len(np.where(triangle_mask)[0])
    nb_quandrangles = len(np.where(quadrangles_mask)[0])

    with open(filename, 'w') as f:

        f.write('# vtk DataFile Version 4.0\n')
        f.write('vtk file generated by meshmagick on %s\n' % time.strftime('%c'))
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS %u float\n' % nv)

        for vertex in vertices:
            f.write('%f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

        f.write('POLYGONS %u %u\n' % (nf, 4*nb_triangles+5*nb_quandrangles))

        for face in faces:
            if face[0] == face[-1]:  # Triangle
                f.write('3 %u %u %u\n' % (face[0], face[1], face[2]))
            else:  # Quadrangle
                f.write('4 %u %u %u %u\n' % (face[0], face[1], face[2], face[3]))

    print('done ...')

# def read_bdf_v0(file_name,verbose=False, keys={'nCh':'signals_in_file', 'readSignal':'readSignal', 'fs':'getSampleFrequency(0)','ch_label':'getSignalLabels()',  'ch_header':'getSignalHeaders()', 'patient':'patient','gender':'getGender()','duration':'getFileDuration()'}):
#
#     fileObj = pyedflib.EdfReader(file_name)
#     nCh = eval(f"fileObj.{keys['nCh']}")
#     X   = eval(f"np.array([fileObj.{keys['readSignal']}(i) for i in range(nCh)])")
#     fs  = eval(f"fileObj.{keys['fs']}")
#     Dur = eval(f"fileObj.{keys['duration']}")
#     ch_names   = eval(f"fileObj.{keys['ch_label']}")
#     Sigheaders = eval(f"fileObj.{keys['ch_header']}")
#     patient    = eval(f"fileObj.{keys['patient']}")
#     patient_id = patient.decode().strip().capitalize()
#     gender     = eval(f"fileObj.{keys['gender']}")
#     fileObj.close()
#     if verbose: print(X.shape)
#     info = {'fs': fs,'Duration':Dur,'nCh':nCh, 'ch_names':ch_names,
#             'Signal_headers':Sigheaders, 'patient':patient,'patient_id':patient_id}
#     return X, info


def read_bdf(file_name,verbose=False,readSignal='readSignal',nCh = None,fs_meth = 'getSampleFrequency',
                                         attrs = {'nCh':'signals_in_file','patient':'patient'},
                                         methds = {'ch_labels':'getSignalLabels',
                                                   'ch_header':'getSignalHeaders', 'gender':'getGender',
                                                   'duration':'getFileDuration'}):

    fileObj = pyedflib.EdfReader(file_name)
    info = {}
    if verbose: print('Reading file ...')
    try:
        fs =  getattr(fileObj, fs_meth)(0)
        info['fs'] = fs
        if verbose: print('fs',fs,sep=':\t')
    except:
        fs=None

    for key in attrs:
        try:
            value =  getattr(fileObj, attrs[key])
            info[key] = value
            if verbose: print(key,value,sep=':\t')
        except:
            print('bad attr : ',key)

    for key in methds:
        try:
            value =  getattr(fileObj, methds[key])()
            info[key] = value
            if verbose:
                if isinstance(value, list):
                    if  verbose>1:
                        print(key,value,sep=':\t')
                else:
                    print(key,value,sep=':\t')
        except:
            print('bad method : ',key)

    if 'patient' in info:
        info['patient_id'] = info['patient'].decode().strip().capitalize()

    ch_labels = None
    if 'ch_labels' in info:
        ch_labels = info['ch_labels']

    readSignal =  getattr(fileObj, readSignal)
    if nCh is None:
        try:
            nCh = info['nCh']
        except:
            try:
                nCh = len(info['ch_label'])
            except:
                try:
                    nCh = len(info['ch_header'])
                except:
                    print("Can't find the number of channels info! either pass is as 'nCh' or correct the attr, methods")

    X = np.array([readSignal(i) for i in range(nCh)])
    fileObj.close()
    return X,fs,ch_labels,info
