# -*- coding: utf-8 -*-

from mnist import MNIST
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import timeit

np.random.seed(42)


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    return X_train[1:], labels_train[1:], X_test, labels_test

X_train, labels_train, X_test, labels_test = load_dataset()

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = []

class KMeans:
    def __init__(self, data, k, method):
        self.data = data
        self.k = k
        self.d = self.data.shape[1]
        if method == 'random':
            self.clusters = [Cluster(np.random.rand(self.d)) for i in range(self.k)]
        elif method == '++':
            print('k-means++ initialization started')
            random_point = self.data[np.random.randint(0,self.data.shape[0])]
            self.clusters = [Cluster(random_point)]
            count = 1
            
            while count < k:
                print('Find center No.' + str(count))
                ranges = []
                for i in data:
                    min_distance, _ = self._find_center(i)
                    ranges.append(min_distance)
                ranges = ranges/np.sum(ranges)
                index = np.argmax(np.random.multinomial(1,ranges))
                self.clusters.append(Cluster(self.data[index]))
                count += 1
                
            
        
    def _find_center(self, point):
        min_distance = np.sum((point - self.clusters[0].center)**2)
        min_index = 0
        
        for i in range(1,len(self.clusters)):
            current_distance =  np.sum((point - self.clusters[i].center)**2)
            if current_distance < min_distance:
                min_distance = current_distance
                min_index = i
         
        return min_distance, min_index
    
    
    def get_centers(self):
        return [i.center for i in self.clusters]
    
    def allocate(self):
        # empty containers
        for i in self.clusters:
            i.points = []
        
        loss = 0
        for i in self.data:
            distance, target_index = self._find_center(i)
            loss += distance
            self.clusters[target_index].points.append(i)
            
        return loss

    def move(self):
        for i in self.clusters:
            if i.points:
                i.center = np.mean(i.points, axis=0)
            else:
                i.center = self.data[np.random.randint(0,self.data.shape[0])]
    
    def is_equal(self,a,b):
        return [tuple(i) for i in a] == [tuple(j) for j in b]
    
    def train(self):
        print('Training started!')
        step = 0
        start_time = timeit.default_timer()
        pre_centers = self.get_centers()
        
        steps = []
        losses = []
        while True:
            step += 1
            loss = self.allocate()
            self.move()
            current_centers = self.get_centers()
            
            if step % 2 == 0:
                steps.append(step)
                losses.append(loss)
                
            if step % 20 == 0:
                print('Current element count per cluster: ', end='')
                print([len(i.points)  for i in self.clusters])
                total_time = timeit.default_timer() - start_time
                print('This is step {:g}, time per step {:.4f}'.format(
                        step, total_time/step))
            
            if self.is_equal(current_centers, pre_centers):
                print('Training complete! Total time {:.4f} and steps {:g}'.format(
                        timeit.default_timer() - start_time, step))
                steps.append(step)
                losses.append(loss)
                return steps,losses
            
            pre_centers = current_centers

      
    
    def display_center(self):
        size = int(np.sqrt(self.d))
        for i in self.clusters:
            img = np.resize(i.center, (size,size))
            plt.imshow(img, cmap='gray')
            plt.show()


# example usage
solver = KMeans(X_train, 5, 'random')
steps,losses = solver.train()
solver.display_center()

plt.plot(steps,losses)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('random initiation and k = 5 ')
plt.show()

solver = KMeans(X_train, 10, 'random')
steps,losses = solver.train()
solver.display_center()

