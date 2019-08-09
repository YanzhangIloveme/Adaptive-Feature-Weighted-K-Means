#coding : utf-8
#Author : Yan Zhang
#Version: 1.0

import numpy as np
import pandas as pd

class Afw_kmeans():

    def __init__(self,data,k) ->(pd.DataFrame,int):
        '''
        :param data: input your data here
        :param k: input your k here
        '''
        self.df     = pd.DataFrame(data)
        self.k      = k
        self.w      = np.array([1/self.df.shape[1]]*self.df.shape[1])
        self.C      = None
        self.v      = self.df.std()
        self.mean   = self.df.mean()
        self.weight_track = []
        self.container = None



    def initializer(self,v):
        '''find initial k-centroids'''
        # mean = self.df.mean()
        # v    = self.df.std()
        mean   = self.mean
        C = []
        if (self.k % 2) == 0:
            for j in range(1,int(self.k/2)+1):
                C.append(mean-2*v/self.k * j)
                C.append(mean+2*v/self.k * j)
        else:
            C.append(mean)
            for j in range(1,(self.k//2)+1):
                C.append(mean - 2 * v / (self.k-1) * j)
                C.append(mean + 2 * v / (self.k-1) * j)
        self.C = np.array(C)
        #return np.array(C)

    def loss(self,centroid,unit):
        '''weighted loss function'''
        unit  = np.array(unit)
        d     = np.sqrt(np.sum(np.multiply((centroid-unit)**2,self.w)))
        return d

    def fit(self,max_iter = 1000):
        iter = 0
        self.initializer(self.v)
        C = self.C
        def assign(C,unit):
            unit = np.array(unit)
            losses = []
            for pos in range(self.k):
                losses.append(self.loss(C[pos],unit))
            return np.argmin(losses)

        while iter < max_iter:
            iter += 1
            container = {}
            labels    = []
            for classes in range(self.k):
                container[classes] = []
            for i in range(len(self.df)):
                val  = self.df.iloc[i]
                __class = assign(self.C,val)
                container[__class].append(np.array(val))
                labels.append(__class)
            #return container
            #try:
            C_new = np.array([np.mean(container[i],axis=0) for i in range(self.k)])
            print('loop_%s Done' %(iter))
            # print('C_new:-----',C_new)
            # print('C:',self.C)
            # print(container)
            #update C
            try:
                if (self.C == C_new).all():
                    break
                else:
                    self.C = C_new
            except:
                if self.C == C_new:
                    break
                else:
                    self.C = C_new
                # shift factors are too large, update the initial value
            check = False
            for classes in range(self.k):
                if len(container[classes])==0:
                    check=True
            if check:
                print('modify model at loop_%s' %(iter))
                self.v = self.v/2
                self.initializer(self.v)

            #update weight
            withinsum = np.array(np.zeros((1, self.df.shape[1])))
            totalavg  = np.array(np.zeros((1, self.df.shape[1])))
            for group in range(self.k):
                withinsum = withinsum + np.sum((container[group]-np.mean(container[group],axis=0))**2,axis=0)
                totalavg  = totalavg  + np.array((np.mean(container[group],axis=0)-self.df.mean())**2)
            #calculate importance factor and update weights based on importance factor
            value  = totalavg/withinsum
            self.weight_track.append(self.w)
            self.w = value/value.sum()

        #find labels
        #dict_new = {value: key for key, value in container.items()}
        self.container  = [(v,k) for v,k in zip(container.values(), container.keys())]
        return labels

    def plot(self):
        if self.weight_track:
            track = pd.DataFrame()
            for i in range(self.df.shape[1]):
                path = [self.weight_track[0][i]]
                for c in range(1, len(self.weight_track)):
                    path.append(self.weight_track[c][0][i])

                track[str(i)] = path
            columns = self.df.columns.values
            track['round'] = [i + 1 for i in range(len(self.weight_track))]

            f = plt.figure(figsize=(10, 6))
            for i in range(x.shape[1]):
                plt.plot(track['round'], track[str(i)], label='%s' % (columns[i]))
            plt.legend()
            plt.title('feature_importance')
            plt.show()

    # coming soon



#%% test

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()
    x    = iris.data
    y    = iris.target

    model = Afw_kmeans(x,3)
    result = model.fit()

# visulize
    vis = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
    vis['y_'] = result
    vis.columns = ['x1','x2','x3','x4','y','y_']
    #map prediction and labels into the same space
    def revis(x):
        if x == 1:
            return 0
        elif x == 0:
            return 1
        else:
            return 2

    vis['y_'] = vis['y_'].apply(revis)

#compared with kmeans

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3)
    km.fit(x)
    vis['km'] = km.labels_
    vis['km'] = vis['km'].apply(revis)


    # calculate accuracy
    right,right2 = 0,0
    for _, i in vis.iterrows():
        if i['y'] == i['y_']:
            right+=1
        if i['y'] == i['km']:
            right2+=1
    print('The accuracy of afw_kmeans is %s' %(right/len(vis)))
    print('The accuracy of kmeans is %s' % (right2 / len(vis)))

    # The accuracy of afw_kmeans is 0.9466666666666667
    # The accuracy of kmeans is 0.8933333333333333

    track = pd.DataFrame()
    for i in range(x.shape[1]):
        path = [model.weight_track[0][i]]
        for c in range(1,len(model.weight_track)):
            path.append(model.weight_track[c][0][i])

        track[str(i)] = path

    track['round'] = [i+1 for i in range(len(model.weight_track))]

    f = plt.figure(figsize=(10,6))
    for i in range(x.shape[1]):
        plt.plot(track['round'],track[str(i)],label = 'X_%s' %(i))
    plt.legend()
    plt.title('feature_importance')
    plt.show()

model.plot()