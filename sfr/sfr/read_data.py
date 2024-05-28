"""
Data management, reading from robot measurements to hdf, loading hdf and so on.

Ideas, to be added / changed
- impulse response
- store sampling rate Fs in a variable
- description & units for measurements in header
"""

import numpy as np
import os
import h5py as hdf

def read_hdf(filename, groups = None):
    """
    read data from hdf file
   
    params:
        filename
        groups (default: None) select a specific group

    returns:
        dictionary holding the data
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'
    data = dict()

    with hdf.File(filename, 'r') as f:
        if groups == None: #no group specified, read all
            groups = f.keys()
        for g in groups:
            # print(f[g])
            data.update({ g : dict() })
            for k in f[g].keys():
                data[g].update({ k : np.array(f[g][k]) })
    return data


def write_hdf(data_dict, filename = './data.h5'):
    """ stores dictionary in hdf format
    parameters:
        data_dict   dictionary holding the data
        filename    file name to be stored (default: ./data.h5)
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'

    if os.path.isfile(filename):
        print("! modifying existing file {} !".format(filename))

    with hdf.File(filename, "a") as f:
        group_name = '_'.join(data_dict['rawfile'][0].split('/')[:-1])
        if group_name in f.keys():
            # print("! overwriting group {} !".format(group_name))
            del f[group_name]
        group = f.create_group(group_name)

        data_dict.pop('rawfile',None) # filter rawfile field
        for att in data_dict.keys():
            group.create_dataset(att, data = data_dict[att])

    return filename
