import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def main(train,test,val):
    a = np.genfromtxt(train, delimiter=',')
    m,n = np.shape(a)
    x = a[1:m,0:n-1]
    y = a[1:m,n-1:n]
    
    a1 = np.genfromtxt(test, delimiter=',')
    m1,n1 = np.shape(a1)
    x1 = a[1:m1,0:n1-1]
    y1 = a[1:m1,n1-1:n1]

    a2 = np.genfromtxt(val, delimiter=',')
    m2,n2 = np.shape(a2)
    x2 = a[1:m2,0:n2-1]
    y2 = a[1:m2,n2-1:n2]

    # optimum values are n_estimators=450, max_features=0.7, min_samples_split=2
    
    # first varry opt estimator
    n_est = [50,150,350,450]
    m_feat = [0.1,0.3,0.5,0.7,0.9]
    min_split = [2,4,6,8,10]

    # n_est = [450]
    # m_feat = [0.1]
    # min_split = [2]

    graph1_1 = []
    graph1_2 = []

    for i in n_est:
        rf = RandomForestClassifier(n_estimators=i, oob_score=True, max_features=0.7,min_samples_split=2)         
        rf.fit(x,y)
        graph1_1.append(100*rf.score(x1,y1))
        graph1_2.append(100*rf.score(x2,y2))
    


    plt.figure(1)
    plt.subplot(211)
    plt.plot(n_est,graph1_1,'-g',Label = "Test acc")
    plt.subplot(212)
    plt.plot(n_est,graph1_2,'-r',Label = "Val acc")
    plt.legend(loc="lower right")
    plt.savefig("est")

    graph2_1 = []
    graph2_2 = []

    for i in m_feat:
        rf = RandomForestClassifier(n_estimators=450, oob_score=True, max_features=i,min_samples_split=2)         
        rf.fit(x,y)
        graph2_1.append(100*rf.score(x1,y1))
        graph2_2.append(100*rf.score(x2,y2))
    

    plt.figure(2)
    plt.subplot(211)
    plt.plot(m_feat,graph2_1,'-g',Label = "Test acc")
    plt.subplot(212)
    plt.plot(m_feat,graph2_2,'-r',Label = "Val acc")
    plt.savefig("feat")

    graph3_1 = []
    graph3_2 = []

    for i in min_split:
        rf = RandomForestClassifier(n_estimators=450, oob_score=True, max_features=0.7,min_samples_split=i)         
        rf.fit(x,y)
        graph3_1.append(100*rf.score(x1,y1))
        graph3_2.append(100*rf.score(x2,y2))
    

    plt.figure(3)
    plt.subplot(211)
    plt.plot(min_split,graph3_1,'-g',Label = "Test acc")
    plt.subplot(212)
    plt.plot(min_split,graph3_2,'-r',Label = "Val acc")
    plt.savefig("split")    



main(sys.argv[1],\
     sys.argv[3],\
     sys.argv[2],\
     sys.argv[4])
