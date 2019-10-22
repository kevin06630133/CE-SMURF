import random
import pandas as pd 
import numpy as np
from sklearn import cluster

class CE_sampling:
    def __init__(self, positive, negative, f=0, r=1, k=5, h=10, c=3):
        self.positive = positive
        self.negative = negative
        self.kmeans_labels = []
        self.kmeans_cluster = []
        self.colname = list(positive)
        self.f = f
        self.r = r
        self.k = k
        self.h = h
        self.c = c

    def jaccard(self, x, y):
        x = np.asarray(x, np.bool) # Not necessary, if you keep your data
        y = np.asarray(y, np.bool) # in a boolean array already!
        return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())
    
    def kmeans(self, i, data) :
        n_attrs = data.shape[1]
        l = [self.colname[j] for j in range(n_attrs)]
        random.shuffle(l) 
        random_col = l[:random.randint(5, 15)]
        tmp_df = data[random_col]
        tmp_df=(tmp_df-tmp_df.min())/(tmp_df.max()-tmp_df.min())
        tmp_df.fillna(0, inplace=True)
        kmeans_train_data = np.array(tmp_df)
        kmeans_fit = cluster.KMeans(n_clusters = self.c).fit(kmeans_train_data)
        n_values = np.max(kmeans_fit.labels_) + 1
        
        return np.eye(n_values)[kmeans_fit.labels_], kmeans_fit.labels_ 
        
    def CI_match(self, data) :
        len_data = len(data)
        match_cluster = np.zeros((h, self.c))
        match_cluster[0] = np.array([i for i in range(self.c)])
        kmeans_labels = np.zeros((self.h, len_data, self.c))
        kmeans_cluster = np.zeros((self.h, len_data))
        for i in range(self.h) :
            kmeans_labels[i], kmeans_cluster[i] = self.kmeans(i, data)
        
        for i in range(1, self.h) :
            for j in range(self.c) :
                now_c = kmeans_labels[i,:,j]
                
                jac = [self.jaccard(now_c, kmeans_labels[0,:,k]) for k in range(self.c)]
                cc = jac.index(max(jac))

                match_cluster[i,j] = cc
        
        for i in range(0, self.h) :
            for j in range(len_data) :
                kmeans_cluster[i,j]  = match_cluster[i, int(kmeans_cluster[i,j])]

        CIs = np.zeros((len_data))
        for i in range(len_data) :
            tmp = 0
            maj_c = np.bincount(np.asarray(kmeans_cluster[:,i], np.int)).argmax()
            for j in range(0, self.h) :
                if maj_c ==  int(kmeans_cluster[j,i]) :
                    tmp += 1
            CIs[i] = tmp / self.h
            
        boundary = []
        center = []
        
        avg_CI = np.mean(CIs)
        for i, CI in enumerate(CIs) :
            if CI >= avg_CI :
                center.append(i)
            else :
                boundary.append(i)
        
        return data.iloc[center], data.iloc[boundary]
    
    def sampling(self):
        #==============================================
        #Positive data + CE-SMOTE
        #==============================================
        if self.f != 0 :
            p_center, p_boundary = self.CI_match(self.positive)
            sm = self.Smote(p_boundary, N=self.f*100, k=self.k)
            over_positive = sm.over_sampling()
        else :
            p_center = self.positive
            over_positive = pd.DataFrame(columns=self.colname)
            
        #==============================================
        #Negative data + CE-Under
        #==============================================
        if self.r != 1.0 :
            n_center, n_boundary = self.CI_match(self.negative)
            random_negative = self.random_drop(n_center, int(len(n_center)*self.r))
        else :
            n_boundary = pd.DataFrame(columns=self.colname)
            random_negative = self.negative
        
        #==============================================
        #Data Merge
        #==============================================
        #print(len(random_negative), len(over_positive))
        keep = pd.concat([p_center, n_boundary], axis=0)
        keep_label = np.hstack((np.ones(len(p_center)), np.zeros(len(n_boundary))))
        X = np.concatenate((np.array(keep), np.array(random_negative), np.array(over_positive)), axis=0)
        y = np.hstack((keep_label, np.zeros(random_negative.shape[0]), np.ones(over_positive.shape[0])))
        
        return X, y
