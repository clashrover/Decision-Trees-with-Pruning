import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def main(train,test,val):
    n_est = [50,150,350,450]
    m_feat = [0.1,0.3,0.5,0.7,0.9]
    min_split = [2,4,6,8,10]
    a = np.genfromtxt(train, delimiter=',')
    m,n = np.shape(a)
    x = a[1:m,0:n-1]
    y = a[1:m,n-1:n]
    opt_rf = None
    optn_est = None
    optm_feat = None
    optmin_split = None
    oobscore = float("-inf")
    
    for i in n_est:
        for j in m_feat:
            for k in min_split:
                rf = RandomForestClassifier(n_estimators=i, oob_score=True, max_features=j,min_samples_split=k)         
                rf.fit(x,y)
                if rf.oob_score_ > oobscore:
                    opt_rf = rf
                    optn_est = i
                    optm_feat = j
                    optmin_split = k
                    oobscore = rf.oob_score_
                    
    
    print("Opt est:",optn_est,"\nOpt # features:",optm_feat,"\nOpt # split:",optmin_split)
    a1 = np.genfromtxt(test, delimiter=',')
    m1,n1 = np.shape(a1)
    x1 = a[1:m1,0:n1-1]
    y1 = a[1:m1,n1-1:n1]

    a2 = np.genfromtxt(val, delimiter=',')
    m2,n2 = np.shape(a2)
    x2 = a[1:m2,0:n2-1]
    y2 = a[1:m2,n2-1:n2]

    print("train Acc:", 100*opt_rf.score(x,y))
    print("Oob Score",100*oobscore)
    print("test Acc:", 100*opt_rf.score(x1,y1))
    print("val Acc:", 100*opt_rf.score(x2,y2))
                    



main(sys.argv[1],\
     sys.argv[3],\
     sys.argv[2],\
     sys.argv[4])

