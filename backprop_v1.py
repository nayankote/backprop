import numpy as np
import random
import mnist_loader

class Network:
    def __init__(self,X,y,num_units):
        self.X = X
        self.y = y
        self.weights = [np.random.randn(w2,w1) for w2,w1 in zip(num_units[1:],num_units[:-1])]
        self.biases = [np.random.randn(b,1) for b in num_units]
        self.num_layers = len(num_units)
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        for e in range(epochs):
            training_data = random.shuffle(training_data)
            mini_batches = [training_data[k,k+1] for k in range(0,len(training_data),mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
                
            print("Epoch {0} training done.".format(e))
            if test_data:
                print("Epoch {0} : Accuracy {1}".format(e,self.evaluate(test_data)))
        return
    
    def evaluate(self, test_data):
        res=[]
        for x,y in test_data:
            res.append(np.argmax(self.feed_forward_full(x)) == y)
        return sum(res)/len(res)
        
    def feed_forward_full(self,x):
        a = x
        for w,b in zip(self.weights, self.biases):
            a = np.dot(w, a) + b
        return a
        
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)
            nabla_w = [nw + ndw for nw,ndw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nb + ndb for nb,ndb in zip(nabla_b,delta_nabla_b)]
        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        return 
    
    def backprop(self,x,y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        activations = [x]
        zs = []
        
        for w,b in zip(self.weights, self.biases):
            zs.append(np.dot(w,activations[-1]) + b)
            activations.append(self.sigma(zs[-1]))
            
        deltas = []
        deltas.append(np.multiply(self.cost_derivative(activations[-1],y),self.sigma_dash(zs[-1])))
        nabla_b[-1] = deltas[-1]
        nabla_w[-1] = np.dot(deltas[-1],activations[-2].T)
        
        for l in range(2,self.num_layers):
            deltas.append(np.multiply(np.dot(self.weights[-l+1].T,deltas[-l+1]),
                                      self.sigma_dash(zs[-l])))
            nabla_b[-l] = deltas[-l]
            nabla_w[-l] = np.dot(deltas[-l],activations[-l-1].T)
            
        return nabla_w, nabla_b
    
    def sigma(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigma_dash(self,z):
        return self.sigma(z).*(1 - self.sigma(z))
    
    def cost_derivative(self,a,y):
        return a-y