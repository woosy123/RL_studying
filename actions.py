import os
import os.path
import logging
import pandas as pd
SUDO_PASSWORD = 'password'

# max number of COSs in Intel CAT
# check by executing 'pqos -s'
NUM_COS_CAT = 4
NUM_COS_MBA = 8
NUM_NET_CLS = 16


metrics_path = 'metrics.txt'
metrics = pd.read_csv(metrics_path, sep=' ',header=None, names=['val'])

result_path = 'result.txt'

def cpu(id, value, period=100000):
    if value < 0:
        raise ValueError('Invalid Value!')
    
    original = float(metrics['val'][0][:5])   
    curr_value = str(int(original) + int(value))
            
    with open(result_path, "r+") as f:
        data = f.read()
        f.seek(0)
        f.write(curr_value)
        f.write('\n')
        f.truncate()

def memory(id, value, cores = 4):
    if value < 0 or value > NUM_COS_MBA-1:
        raise ValueError('Invalid Value!')
    
    with open(result_path, "r+") as f:
        data = f.read()
        f.write(str(value)) # + ' '.join(cores)
        f.write('\n')
        f.truncate()

def network(id, value):
    if value < 0 or value > NUM_NET_CLS-1:
        raise ValueError('Invalid Value!')
    value = str(value)
    with open(result_path, "r+") as f:
        data = f.read()
        f.write('0x001000' + value.zfill(2))
        f.truncate()

