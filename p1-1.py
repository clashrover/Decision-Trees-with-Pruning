import numpy as np
import sys
import math

class Node():
    """
    Node of Decision tree
    """
    def __init__(self,data):
        self.data = data
        self.label = None
        self.children = []
        self.attribute = None

    def isLeaf(self):
        if len(self.children)==0:
            return True

    def addChild(self,n):
        self.children.append(n)

    def getData(self):
        return self.data

    def setAttr(self,attr):
        self.attribute = attr

    def setVal(self,val):
        self.val = val

    def setLabel(self,a):
        self.label = a
      
    def getLabel(self):
        if self.label != None:
            return int(self.label)
        x,y = self.data
        (labels, counts) = np.unique(y[:,0],return_counts=True)
        return int(labels[np.argmax(counts)])
        
def entropy(D):
    x,y = D
    e = 0
    (labels, counts) = np.unique(y[:,0],return_counts=True)
    s = np.sum(counts)
    for i in range(len(labels)):
        p = counts[i]/s
        e -= p*math.log2(p)
    return e

def split(D,xj,val=None):
    # we split the data D on the basis of val of xj
    # all rows with xj value > val on right side and rest on left
    dataList = []
    x,y = D
    v = None
    if xj<10: # means continuous
        # find median of xj in data D
        median = int(np.median(x[:,xj]))
        v = median 
    else:
        # split on 0,1
        v=0
    if val != None:
        v= val
    dtemp = np.hstack((x,y))
    dleft = dtemp[dtemp[:,xj]<=v]
    ml,nl = np.shape(dleft)
    xl,yl = dleft[:,0:nl-1], dleft[:,nl-1:]
    
    dright = dtemp[dtemp[:,xj]>v]
    mr,nr = np.shape(dright)
    xr,yr = dright[:,0:nr-1], dright[:,nr-1:]
    
    if ml >0:
      dataList.append((xl,yl))
    if mr >0:
      dataList.append((xr,yr))

    return dataList, v


def chooseAttributeAndSplit(D):
    # find the entropy after split for each attribute and return the one that minimises it
    x,y = D
    m,n = np.shape(x)
    minEntropy = float("inf")
    minAttr = None
    final_dataList = None
    final_value = None
    for i in range(n):
        # now split on basis of attribute i
        data_list, value = split(D,i) 
        if len(data_list) < 2:
            continue
        total_entropy = 0
        for d in data_list:
            a,b = d
            a1,b1 = np.shape(a)
            a1 = a1/m
            total_entropy += a1*entropy(d)
        
        if minEntropy > total_entropy:
            minEntropy = total_entropy
            minAttr = i
            final_dataList = data_list
            final_value = value
    
    return minAttr, final_dataList, final_value

def checkLeaf(D):
    x,y = D
    if np.max(y[:,0]) == np.min(y[:,0]):
        return int(y[0][0])
    return -1

class decisionTree():
    def __init__(self):
        self.currentSize = 0
        self.root = None
        self.queue = None

    def getRoot(self):
        return self.root

    def createLeaf(self,label):
        n = Node(None)
        n.setLabel(label)
        return n
    
    def growTree(self,D,size=float("inf")):
        q = []
        root = None
        l = checkLeaf(D)
        if l>-1:
            root = self.createLeaf(l)
        else:
            root = Node(D)
            q.append(root)

        self.currentSize+=1
        self.root = root
        while(len(q)>0):
            r = q.pop(0)
            xbest, dL, vl = chooseAttributeAndSplit(r.getData())
            # print("best attr, val, size of tree:", xbest, vl, self.currentSize)
            r.setAttr(xbest)
            r.setVal(vl)
            for i in range(len(dL)):
                n = None
                a = checkLeaf(dL[i])
                if a>-1:
                    n = self.createLeaf(a)
                else:
                    n = Node(dL[i])
                    q.append(n)
                r.addChild(n)
                self.currentSize+=1
            
            if self.currentSize >= size:
                break
        self.queue = q


    def updateTree(self,stepSize=1):
        q = self.queue
        if len(q)==0:
            return -1
        counter=0
        while(len(q)>0):
            if counter>=stepSize:
                break
            r = q.pop(0)
            xbest, dL, vl = chooseAttributeAndSplit(r.getData())
            r.setAttr(xbest)
            r.setVal(vl)
            for i in range(len(dL)):
                n = None
                a = checkLeaf(dL[i])
                if a>-1:
                    n = self.createLeaf(a)
                else:
                    n = Node(dL[i])
                    q.append(n)
                r.addChild(n)
                self.currentSize+=1
                counter+=1

        return 1

    def infer(self,root,D):
        x,y = D
        # print(1)
        score = 0
        if root.isLeaf():
            # print("in leaf")
            l = 1.0*root.getLabel()
            # print(l)
            m,n = np.shape(x)
            # score = m
            score = np.count_nonzero(y[:,0] == l)
            return score
        
        dL,v = split(D,root.attribute,root.val)
        for i in range(len(dL)):
            x,y = dL[i]
            score += self.infer(root.children[i],dL[i])
        
        return score


def main(train,test,val):
    # read data into numpy array
    a = np.genfromtxt(train, delimiter=',')
    m,n = np.shape(a)
    x = a[1:m,0:n-1]
    y = a[1:m,n-1:n]
    m,n = np.shape(x)
    sizeL = []
    accL = []
    # form the decision tree model
    dTree = decisionTree()
    dTree.growTree((x,y),1)
    print("Tree made of size: ", dTree.currentSize)

    a1 = np.genfromtxt(test, delimiter=',')
    m1,n1 = np.shape(a1)
    x1 = a1[1:m1,0:n1-1]
    y1 = a1[1:m1,n-1:n1]
    m1,n1 = np.shape(x1)
    tsizeL = []
    taccL  = []

    a2 = np.genfromtxt(val, delimiter=',')
    m2,n2 = np.shape(a2)
    x2 = a2[1:m2,0:n2-1]
    y2 = a2[1:m2,n-1:n2]
    m2,n2 = np.shape(x2)
    vsizeL = []
    vaccL  = []
    for i in range(45):
        score = dTree.infer(dTree.getRoot(),(x,y))
        acc = score/m
        print("train Acc:",acc)
        sizeL.append(dTree.currentSize)
        accL.append(acc)

        score = dTree.infer(dTree.getRoot(),(x1,y1))
        acc = score/m1
        print("test Acc:",acc)
        tsizeL.append(dTree.currentSize)
        taccL.append(acc)

        score = dTree.infer(dTree.getRoot(),(x2,y2))
        acc = score/m2
        print("Val Acc:",acc)
        vsizeL.append(dTree.currentSize)
        vaccL.append(acc)
        
        dTree.updateTree(2000)
        print("update",i,":",dTree.currentSize)
    

    import matplotlib.pyplot as plt
    plt.plot(sizeL,accL,label = "TRAIN")

    plt.plot(tsizeL,taccL,label = "TEST")

    plt.plot(vsizeL,vaccL,label = "VAL")
    plt.savefig('Acc vs Size p11.png')

    
    
    
# main(sys.argv[1],sys.argv[2],sys.argv[3])

main(sys.argv[1],\
     sys.argv[3],\
     sys.argv[2],\
     sys.argv[4])



    

# For reference
# Elevation:Continuous,
# Aspect:Continuous,
# Slope:Continuous,
# Horizontal_Distance_To_Hydrology:Continuous,
# Vertical_Distance_To_Hydrology:Continuous,
# Horizontal_Distance_To_Roadways:Continuous,
# Hillshade_9am:Continuous,
# Hillshade_Noon:Continuous,
# Hillshade_3pm:Continuous,
# Horizontal_Distance_To_Fire_Points:Continuous,
# Wilderness_Area_1:Discrete,
# Wilderness_Area_2:Discrete,
# Wilderness_Area_3:Discrete,
# Wilderness_Area_4:Discrete,
# Soil_Type_1:Discrete,Soil_Type_2:Discrete,Soil_Type_3:Discrete,Soil_Type_4:Discrete,Soil_Type_5:Discrete,Soil_Type_6:Discrete,Soil_Type_7:Discrete,Soil_Type_8:Discrete,Soil_Type_9:Discrete,Soil_Type_10:Discrete,Soil_Type_11:Discrete,Soil_Type_12:Discrete,Soil_Type_13:Discrete,Soil_Type_14:Discrete,Soil_Type_15:Discrete,Soil_Type_16:Discrete,Soil_Type_17:Discrete,Soil_Type_18:Discrete,Soil_Type_19:Discrete,Soil_Type_20:Discrete,Soil_Type_21:Discrete,Soil_Type_22:Discrete,Soil_Type_23:Discrete,Soil_Type_24:Discrete,Soil_Type_25:Discrete,Soil_Type_26:Discrete,Soil_Type_27:Discrete,Soil_Type_28:Discrete,Soil_Type_29:Discrete,Soil_Type_30:Discrete,Soil_Type_31:Discrete,Soil_Type_32:Discrete,Soil_Type_33:Discrete,Soil_Type_34:Discrete,Soil_Type_35:Discrete,Soil_Type_36:Discrete,Soil_Type_37:Discrete,Soil_Type_38:Discrete,Soil_Type_39:Discrete,Soil_Type_40:Discrete,
# Cover_Type:Class