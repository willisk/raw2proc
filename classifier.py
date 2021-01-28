import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Model():
    
    def __init__(self, net, trainloader, testloader):
        
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        
        #to analyse training
        self.train_err = []
        self.acc_train = []
        self.acc_test = []
        
        self.stamp = str(time.time() - int(time.time()))[2:7]
        
    def train(self, epochs=1, lrs=[1e-3],plot=False):
        
        self.acc_train.append(self.predict(self.trainloader))
        self.acc_test.append(self.predict(self.testloader))
        
        print(self.acc_train)
        print(self.acc_test)
        
        for j in range(len(lrs)):
            
            optimizer = optim.SGD(self.net.parameters(), lr=lrs[j], momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):  

                running_loss = 0.0
                running_loss_ep = 0.0
                for i,data in enumerate(self.trainloader, 0):
                    inputs, labels = data
                    labels = torch.squeeze(labels,dim=-1)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    outputs = self.net(inputs)
                    '''
                    print('out')
                    print(type(outputs))
                    print(outputs.shape)
                    print(outputs.dtype)
                    print('in')
                    print(type(labels))
                    print(labels.shape)
                    print(labels.dtype)
                    '''
                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    running_loss_ep += loss.item()
                    if i%40==0:
                        print('epoch %i minimbatch %i --  loss: %.8e -- learning rate: %.8e' % (epoch + 1, i+1, running_loss/40,lrs[j]))
                        running_loss = 0.0
                if plot:  
                    self.train_err.append(running_loss_ep/(i+1))
                    self.acc_train.append(self.predict(self.trainloader))
                    self.acc_test.append(self.predict(self.testloader))
                print('EPOCH %i DONE --  loss: %.8e' % (epoch + 1, running_loss_ep/(i+1)))
                print(self.acc_train)
                print(self.acc_test)
                
        if plot:
            self.plot_loss(lrs, epochs)
            
        print('Finished Training')
        
        return 
    
    def plot_loss(self,lrs, epochs):
        
        fig = plt.figure(figsize=(16,4))
        X = np.arange(len(self.train_err))
        Y = np.arange(len(self.acc_train))
        ax1 = fig.add_subplot(121)
        ax1.title.set_text('Training error over %i epochs with learning rates %s'%(epochs,str(lrs)))
        ax1.plot(X,self.train_err)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('CrossEntropyLoss',rotation='vertical')

        ax2 = fig.add_subplot(122)
        ax2.title.set_text('Accuracy on trainset and testset over %i epochs with learning rates %s'%(epochs,str(lrs)))
        ax2.plot(Y,self.acc_train,label='accuracy on trainset')
        ax2.plot(Y,self.acc_test,label='accuracy on testset')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('Accuracy',rotation='vertical')
        plt.legend()
        fig.savefig('./plots/%s.jpg'%(self.stamp))
        plt.show()
            
        return  
    
    def predict(self, dataloader):
        correct = 0
        total = len(dataloader)*dataloader.batch_size
        self.net.eval()
        with torch.no_grad():
            for data in dataloader:
                inputs,labels = data
                labels = torch.squeeze(labels,dim=-1)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
        self.net.train()      
        return correct / total