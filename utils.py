import pickle
import difflib
import numpy as np

def sparse_tuple_from(sequences, dtype=np.int32):
    ''' create a sparse representention for ctc loss.
    '''
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), \
            np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
  
def write_list_to_file(file_path, lines, mode):
    ''' write a list to file
    '''
    with open(file_path, mode, encoding='utf-8') as fout:
        for line in lines:
            fout.write(line+'\n')


def invert_dict(d):
    ''' invert a dict
      key -> value, value -> key
    '''
    return dict((v,k) for k,v in d.items())
  
  
def write_to_pkl(file_path, data):
    ''' write data into a  pickle file
    '''
    with open(file_path, 'wb') as fin:
        pickle.dump(data, fin, pickle.HIGHEST_PROTOCOL)
  
def load_from_pkl(file_path):
    ''' load data from pickle file
    '''
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
