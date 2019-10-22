import time
import random
import pandas as pd 
import numpy as np
import CE_sampling
import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  

class CE_SMURF :

    def __init__(self, n=10, f=0, r=1, k=5, h=10, t=100, c=3) :
        self.n = n
        self.f = f
        self.r = r
        self.k = k
        self.h = h
        self.t = t
        self.c = c

    def split_data_folds(self, data,split_num):
        shuffled_indices = np.random.permutation(len(data))
        data_resto = len(data) % split_num
        num_per_fold = int(len(data) / split_num)
        data_folds_list = [None for i in range(split_num)]
        data_folds_num = [num_per_fold for i in range(split_num)]
        if data_resto > 0 :
            for i in range(split_num, split_num-data_resto, -1) :
                data_folds_num[i-1] += 1         
        count = 0  
        for i in range(split_num) : 
            data_folds_list[i] = data.iloc[shuffled_indices[count:count+data_folds_num[i]]]
            count += data_folds_num[i]
        
        return data_folds_list

    def split_negative_list(self, data, split_num):
        shuffled_indices = np.random.permutation(len(data))
        data_list = [None for i in range(split_num)]
        part_num = int(len(data) / split_num)
        for i in range(split_num) :
            if i == split_num - 1 :
                data_list[i] = data.iloc[shuffled_indices[part_num*i:]]
            else :
                data_list[i] = data.iloc[shuffled_indices[part_num*i:part_num*(i+1)]]
        
        return data_list

    def random_drop(self, data, num) : 
        shuffled_indices = np.random.permutation(len(data))
        
        return data.iloc[shuffled_indices[:num]]

    def train(self, positive, negative):
        negative_list = self.split_negative_list(negative, self.n)
        rf_list = [None for i in range(self.n)]

        for i in range(self.n):
            labels = np.hstack((np.ones(positive.shape[0]), np.zeros(negative.shape[0])))
            CE = self.CE_sampling(positive, negative, f=self.f, r=self.r, k=self.k, h=self.h, c=self.c)
            X, y = CE.sampling()
            forest = RandomForestClassifier(n_estimators = self.t, max_features = "sqrt")
            rf_list[i] = forest.fit(X, y)
            
        return rf_list

    def test(self, rf_list, test_data):
        score = pd.DataFrame([0 for i in range(len(test_data))], index=test_data.index.get_values(), columns=["Scores"])
        for rf in rf_list :
            score += pd.DataFrame(rf.predict_proba(test_data)[:,1], index=test_data.index.get_values(), columns=["Scores"])
        score = score / len(rf_list)
        
        return score

    def cross_vaildation(self, data_positive, data_negative, n_fold=10) :
        
        all_data = pd.concat([data_positive, data_negative], axis=0)
        answer = np.hstack((np.ones(len(data_positive)), np.zeros(len(data_negative))))
        scores = pd.DataFrame([None for i in range(len(all_data))], index=all_data.index.get_values(), columns=["Scores"])

        data_negative_folds = self.split_data_folds(data_negative, n_fold)
        data_positive_folds = self.split_data_folds(data_positive, n_fold)          
        for fold in range(n_fold):
            ind_positive_test = data_positive_folds[fold]
            ind_negative_test = data_negative_folds[fold]

            ind_all_test = pd.concat([ind_positive_test, ind_negative_test], axis=0)
            ind_positive_train = 0
            ind_negative_train = 0
            count = 0
            for j in range(n_fold) :
                if j == fold :
                    continue
                if count == 0 :
                    ind_positive_train = data_positive_folds[j]
                    ind_negative_train = data_negative_folds[j]
                else :
                    ind_positive_train = pd.concat([ind_positive_train, data_positive_folds[j]], axis=0)
                    ind_negative_train = pd.concat([ind_negative_train, data_negative_folds[j]], axis=0)
                count += 1

            print("Starting training on fold {0}\n".format(str(fold+1)))
            rf_list = self.train(ind_positive_train, ind_negative_train)

            print("Starting test on Fold {0}\n".format(str(fold+1)))
            scores["Scores"][ind_all_test.index.get_values()] = self.test(rf_list, ind_all_test)["Scores"]

            print("End test on Fold {0}\n".format(str(fold+1)))
        
        #print(confusion_matrix(answer, np.where(scores["Scores"]>=0.2,1,0), labels=[1, 0]))
        precision, recall, threshold_prc = precision_recall_curve(answer ,scores["Scores"])
        prc = auc(recall,precision) ###计算auc的值
        tpr, fr, threshold_roc = roc_curve(answer ,scores["Scores"])
        roc = auc(tpr,fr) ###计算auc的值
        
        now_time = time.strftime("%H:%M:%S", time.localtime()) 
        print('Data : ({0}, {1})'.format(len(self.data_positive), len(self.data_negative)))
        print('N fold : {0} Feature num : {1}'.format(n_part, len(self.data_positive.columns)))
        print('Time=%s, n=%d, f=%d, r=%0.1f, k=%d, h=%d, t=%d, c=%d\n' % (now_time, self.n, self.f, self.r, self.k, self.h, self.t, self.c))
        print('ROAUC=%0.3f PRAUC=%0.3f' % (roc, prc))
        
        return scores

if __name__ == "__main__":
    #positive
    data_positive = pd.read_csv("HGMD.csv", index_col=0)
    data_positive = data_positive.drop(columns=["chr", "end", "start", "cls"])
    c_list = sorted(data_positive.columns.values.tolist())
    data_positive = data_positive[c_list]

    #negative
    data_negative = pd.read_csv("HGMD_negative.csv", index_col=0)
    #data_negative = data_negative.drop(['chr20_33759639', 'chr4_156124703'], axis=0)
    data_negative = data_negative[c_list]

    ce = CE_SMURF()
    ce.cross_vaildation(data_positive, data_negative, n_fold=10)

