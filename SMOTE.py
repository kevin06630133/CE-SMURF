import random
import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Smote:
    def __init__(self, samples, N=10):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        if len(samples) <= 5 :
            self.k = len(samples) - 1

        self.samples=np.array(samples)
        self.newindex=0

    def over_sampling(self):
        if self.N == 0 :
            return self.samples
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k+1).fit(self.samples)
        
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            
            self._populate(N,i,nnarray[1:])
        
        self.synthetic = np.concatenate((self.samples, self.synthetic), axis=0)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
