# -*- coding: utf-8 -*-
import scipy as sp

class CV:
    '''
    This class implements the generation of several folds to be used in the cross validation
    '''
    def __init__(self):
        self.it=[]
        self.iT=[]

    def split_data(self,n,v=5):
        ''' The function split the data into v folds. Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None        
        '''
        step = n //v  # Compute the number of samples in each fold
        sp.random.seed(1)   # Set the random generator to the same initial state
        t = sp.random.permutation(n)    # Generate random sampling of the indices
        
        indices=[]
        for i in range(v-1):            # group in v fold
            indices.append(t[i*step:(i+1)*step])
        indices.append(t[(v-1)*step:n])
                
        for i in range(v):
            self.iT.append(sp.asarray(indices[i]))
            l = range(v)
            l.remove(i)
            temp = sp.empty(0,dtype=sp.int64)
            for j in l:            
                temp = sp.concatenate((temp,sp.asarray(indices[j])))
            self.it.append(temp)

    def split_data_class(self,y,v=5):
        ''' The function split the data into v folds. The samples of each class are split approximatly in v folds
        Input:
            n : the number of samples
            v : the number of folds
        Output: None
        '''
        # Get parameters
        n = y.size
        C = y.max().astype('int')
       
        # Get the step for each class
        tc = []
        for j in range(v):
            tempit = []
            tempiT = []
            for i in range(C):
                # Get all samples for each class
                t  = sp.where(y==(i+1))[0]
                nc = t.size
                stepc = nc // v # Step size for each class
                sp.random.seed(i)   # Set the random generator to the same initial state
                tc = t[sp.random.permutation(nc)] # Random sampling of indices of samples for class i
                        
                # Set testing and training samples
                if j < (v-1):
                    start,end = j*stepc,(j+1)*stepc
                else:
                    start,end = j*stepc,nc
                tempiT.extend(sp.asarray(tc[start:end])) #Testing
                k = range(v)
                k.remove(j)
                for l in k:
                    if l < (v-1):
                        start,end = l*stepc,(l+1)*stepc
                    else:
                        start,end = l*stepc,nc
                    tempit.extend(sp.asarray(tc[start:end])) #Training

            self.it.append(tempit)
            self.iT.append(tempiT)
                
                
            
        
        
    
