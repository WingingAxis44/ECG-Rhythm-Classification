
import wfdb
import pandas as pd
import numpy as np


abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S','Q']



# list of patients
# Eventually something more modular will be implemented than a hardcoded list
# like this.
pts = ['100', '101', '102', '103', '104', '105', '106', '107',
       '108', '109', '111', '112', '113', '114', '115', '116',
       '117', '118', '119', '121', '122', '123', '124', '200',
       '201', '202', '203', '205', '207', '208', '209', '210',
       '212', '213', '214', '215', '217', '219', '220', '221',
       '222', '223', '228', '230', '231', '232', '233', '234']



#Reads a signal ECG and gets the physical signal (raw data) and associated annotations and symbols
def load_ecg(file):
    record = wfdb.rdrecord(file)
    
    ann = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal
    
    sym = ann.symbol
    samp = ann.sample
    
    
    return p_signal, sym, samp


def make_dataset(data_path,num_sec=1, fs=360):
    
    # global pts
    # global abnormal

    num_cols = (int) (2*num_sec * fs)

    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1),dtype=np.int8)

    max_rows = []
 
    
    for patient in pts:

        file = data_path + patient
   
        p_signal,sym,samp = load_ecg(file)
        
        p_signal = p_signal[:,0]

  
        df_ann = pd.DataFrame({'atr_sample':samp, 'atr_sym':sym})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal + ['N'])]
   
      
        X, Y = build_XY(p_signal,df_ann, num_cols, num_sec, fs)
    


        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)
        

    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]
   
    #print(np.unique(rhythm_all, return_counts=True))
     # check sizes make sense
 
    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert X_all.shape[0] == Y_all.shape[0], 'number of X, Y rows messed up'
   
    return X_all, Y_all  



def build_XY(p_signal, df_ann, num_cols, num_sec, fs):
    

    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows,1),dtype=np.int8)
  
    # keep track of rows
    max_row = 0

  
    
    for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):

            
        left = (int) (max([0,(atr_sample - num_sec*fs) ]))
        right = (int) (min([len(p_signal),(atr_sample + num_sec*fs) ]))
      
        x = p_signal[left: right]
        
         

        if len(x) == num_cols:
            X[max_row,:] = x

            
            Y[max_row,:] = get_class(atr_sym)
            
            max_row += 1
   
    X = X[:max_row,:]
    Y = Y[:max_row,:]
    
    return X, Y


def get_class(sym):
    N = ['N','e','j','L','R']
    S = ['A','J','S','a']
    V = ['E','V']
    F = ['F']
    Q = ['/','Q','f']
    if sym in N: # normal beat
        return 0
    if sym in S: # superventricular beat
        return 1
    if sym in V: # ventricular beat
        return 2
    if sym in F: # fusion beat
        return 3
    if sym in Q: # unknown beat
        return 4