import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from collections import defaultdict
import warnings
from hmmlearn.hmm import GaussianHMM
import time
import timeit
import itertools
from scipy import signal
import skimage.measure
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class SensorHMM(object):
    """
    Author : Sunil Prakash
    """
    
    def __init__(self,name,data_directory, data_key, no_features,n_components=1,params="ct",
                 split_at=26,actions=["a1,a2,a3"]):
        
        self.name=name
        self.n_components = n_components
        self.data_directory = data_directory
        self.data_key = data_key
        self.no_features = no_features
        self.split_at = split_at
        self.actions = actions
        self.all_actions= ["a"+str(num) for num in range(1,28)]
        self.models={}
        self.params=params
        
        self.all_data=None
        self.training_data = None
        self.testing_data = None
        
        self.feature_seq_dict={}
        self.seq_len_dict={}
        
    
    def load_data(self):
        files = os.listdir(self.data_directory)
        all_data = [] 
        for ii, file in enumerate(files, 1):
            if file.endswith(".mat"):
                mat_contents = sio.loadmat(self.data_directory+'/'+file)
                d_skel=mat_contents[self.data_key]
                action = file.split("_")[0]
                all_data.append((d_skel,action))
        print("Data loaded ",len(all_data))
        #self.all_data = all_data
        return all_data

    def split_data(self,all_data,split_at=26, prepare=True):
        self.split_at = split_at
        ddct = defaultdict(list)
        training_data=[]
        testing_data=[]
        for X,y in all_data:
            for action in self.all_actions:
                if(action == y):
                    ddct[action].append(1)
                    if len(ddct[action]) < self.split_at+1:
                        training_data.append((X,y))
                    else:
                        testing_data.append((X,y))
        self.training_data=training_data
        self.testing_data=testing_data
        print("Done splitting", len(training_data), len(testing_data))
        return training_data, testing_data
        #return zip(*training_data), zip(*testing_data)

    def fetch_training_data_by_action(self,records,action):
        action_pairs=[]
        for X,y in records:
            if y == action:
                action_pairs.append((X,y))
        return action_pairs

    def fit_action(self, action, num_hidden_states, features, lengths):

        warnings.filterwarnings("ignore", category=DeprecationWarning)  
        model = GaussianHMM(n_components=num_hidden_states, n_iter=1000,random_state=123,params=self.params).fit(features,lengths)
        logL = model.score(features,lengths)
        return model, logL


    def get_action_and_seq_len_dict(self, training_data,actions):
        all_sequences={}
        all_lengths={}
        
        
        for action in actions:
            X,lengths = self.get_hmm_formatted_features(training_data,action)
            all_sequences[action] = X
            all_lengths[action]=lengths
        return all_sequences,all_lengths


    def fit(self,training_data,actions):
        self.actions = actions
        
        feature_seq_dict,seq_len_dict = self.get_action_and_seq_len_dict(training_data,actions)

        for action in self.actions:
            print("training for ",action)
            X = feature_seq_dict[action]
            X = np.array(X)
            print(X.shape)
            lengths = seq_len_dict[action]
            model, logL = self.fit_action(action, self.n_components, X ,lengths)
            print("ll",logL)
            self.models[action]=model
        print("** training complete **")



    def get_hmm_formatted_features(self, records,action):
        x_contatinated = np.zeros((1,self.no_features))
        lengths=[]
        action_features = []
        if self.name=="skeleton":
            action_features = self.extact_features_skelton(records,action)
        elif self.name=="depth":
            action_features = self.extact_features_depth(records,action)
        else:
            action_features = self.extact_features(records,action)
        #print("length",action_features[0].shape)

        for subject_action in list(action_features):
            lengths.append(subject_action.shape[0])
            x_contatinated = np.append(x_contatinated,subject_action,axis=0)


        x_contatinated = np.delete(x_contatinated, 0, axis=0)
        #print(np.array(x_contatinated).shape)
        #print(lengths)
        return np.array(x_contatinated),lengths
    
    
    def extact_features(self,records,action):
        #print("iner",records)
        all_features,_ = zip(*self.fetch_training_data_by_action(records,action))
        return all_features
    
    def extact_features_depth(self,records,action):
        #print("iner",records)
        x_contatinated = []
        all_features,_ = zip(*self.fetch_training_data_by_action(records,action))
        
        w_k_v = np.array([[3,0,-3],[10,0,-10],[3,0,-3]]) #Sobel–Feldman 
        w_k_h = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])


        for subject_action in list(all_features):
            for i in range(subject_action.shape[2]):
                #print(i)
                lay1_b = subject_action[:,:,i]
                if i==0:
                    plt.imshow(lay1_b)
                for j in range(4):
                    #print(lay1_b.shape, w_k_v.shape,subject_action[:,:,i].shape)
                    lay1v = signal.convolve2d(lay1_b, w_k_v, 'valid')
                    lay1h = signal.convolve2d(lay1_b, w_k_v, 'valid')
                    #print(lay1v.shape)
                    #lay1_c = np.stack([lay1v,lay1h], axis=2)
                    lay1_b = skimage.measure.block_reduce(lay1v, (2,2), np.max)

                #print(subject_action[:,:,i].shape)
                #print(lay1_b.shape)

                #x_contatinated.append(lay1_b.reshape(-1,))
                x_contatinated.append(lay1_b)
        return x_contatinated
        
        
    ## for extracting skeleton feature
    def extact_features_skelton(self,records,action):
        all_features,_ = zip(*self.fetch_training_data_by_action(records,action))
        #print(len(all_features))

        delta_records=[]
        for record in all_features:
            rec_length = record.shape[2]
            diffs=[]
            for idx in range(1,rec_length):
                diff_x = record[:,0,idx] - record[:,0,idx-1]
                diff_y = record[:,1,idx] - record[:,1,idx-1]
                diff_z = record[:,2,idx] - record[:,2,idx-1]
                diffs.append(np.sqrt(np.square(diff_x)+np.square(diff_y)+np.square(diff_z)))
            delta_records.append(np.array(diffs))
        #print(all_features[0].shape)
        #print(np.array(delta_records[0]).shape)
        return np.array(delta_records)

    def show_model_stats(self, word, model):
        print("Number of states trained in model for {} is {}".format(word, model.n_components))    
        variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
        for i in range(model.n_components):  # for each hidden state
            print("hidden state #{}".format(i))
            print("mean = ", model.means_[i])
            print("variance = ", variance[i])
            print()

    import math
    from matplotlib import (cm, pyplot as plt, mlab)
    def visualize(self,word, model):
        """ visualize the input model for a particular word """
        variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
        figures = []
        for parm_idx in range(len(model.means_[0])):
            xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
            xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
            fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
            colours = cm.rainbow(np.linspace(0, 1, model.n_components))
            for i, (ax, colour) in enumerate(zip(axs, colours)):
                x = np.linspace(xmin, xmax, 100)
                mu = model.means_[i,parm_idx]
                sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
                ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
                ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

                ax.grid(True)
            figures.append(plt)
        for p in figures:
            p.show()


    def predict(self,testing_data):
        probabilities = []
        guesses = []
        

        for Xs,ys in [(Xs,ys) for Xs,ys in testing_data if ys in self.actions]:
            X,L = self.get_hmm_formatted_features([(Xs,ys)],ys)

            bestLL = float("-inf")
            bestAction = None
            probs = {}
            for action, model in self.models.items():
                try:
                    ll = model.score(X)
                    if ll > bestLL:
                            bestLL = ll
                            bestAction = action
                            probs[action] = ll
                except:
                    print(" ! ",end=" ")
                    pass

            guesses.append(bestAction)
            probabilities.append(bestLL)
        return guesses,probabilities


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def print_classification_report(self,actual,predicted):
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted, target_names=self.actions))
        cnf_matrix = confusion_matrix(actual, predicted)
        np.set_printoptions(precision=2)
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.actions,
                              title='Confusion matrix')
        
    
    

