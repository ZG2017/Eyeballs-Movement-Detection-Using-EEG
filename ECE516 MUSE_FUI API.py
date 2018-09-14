
# coding: utf-8

# In[96]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import random
import torch
from torch.autograd import Variable
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
pathsave = r"G:\ECE516\FUI" +"\\"


# In[107]:


# use to get a numpy array from .txt file
def eeg_extract(flie_path):
    f = open(flie_path,"r")
    channel1 = []
    channel2 = []
    channel3 = []
    channel4 = []
    for line in f:
        if line.split()[1] == "/muse/eeg":       # change there to get different data
            channel1.append(float(line.split()[3]))
            channel2.append(float(line.split()[4]))
            channel3.append(float(line.split()[5]))
            channel4.append(float(line.split()[6]))
    f.close()
    return np.array([channel1,channel2,channel3,channel4],dtype = np.float32)

# use to plot all four eeg channel 
def plot_eeg(data,          # numpy array: data
             scope_low,     # int: lower boundary of data to plot 
             scope_high):   # int: upper boundary of data to plot
    if scope_high > len(data[0]):
        print("higher boundary out of limits")
        return 
    xxx = range(0,len(data[0]),1)[scope_low:scope_high]
    fig1 = plt.figure(figsize = (20,7))
    ax1 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax1.plot(xxx,data[0][scope_low:scope_high],"r-",label = "channel 1")
    plt.legend(loc = 1,shadow = True)
    plt.show()

    fig2 = plt.figure(figsize = (20,7))
    ax2 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax2.plot(xxx,data[1][scope_low:scope_high],"b-",label = "channel 2")
    plt.legend(loc = 1,shadow = True)
    plt.show()

    fig3 = plt.figure(figsize = (20,7))
    ax3 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax3.plot(xxx,data[2][scope_low:scope_high],"g-",label = "channel 3")
    plt.legend(loc = 1,shadow = True)
    plt.show()

    fig4 = plt.figure(figsize = (20,7))
    ax4 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax4.plot(xxx,data[3][scope_low:scope_high],"y-",label = "channel 4")
    plt.legend(loc = 1,shadow = True)
    plt.show()

    fig5 = plt.figure(figsize = (20,7))
    ax5 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax5.plot(xxx,data[0][scope_low:scope_high],"r-",label = "channel 1")
    ax5.plot(xxx,data[1][scope_low:scope_high],"b-",label = "channel 2")
    ax5.plot(xxx,data[2][scope_low:scope_high],"g-",label = "channel 3")
    ax5.plot(xxx,data[3][scope_low:scope_high],"y-",label = "channel 4")
    plt.legend(loc = 1,shadow = True)
    plt.show()
    
# plot only one channel
def plot_one_channel(data,            # numpy array: data
                     whichchannel,    # int: which channel wanted to plot
                     scope_low,       # int: lower boundary of data to plot 
                     scope_high,      # int: upper boundary of data to plot 
                     reference):      # float: reference number of data
    if scope_high > len(data[0]):
        print("higher boundary out of limits")
        return 
    xxx = range(0,len(data[whichchannel-1]),1)[scope_low:scope_high]    
    fig1 = plt.figure(figsize = (20,7))
    ax1 = plt.subplot()
    plt.title("Data")
    plt.xlabel("Time")
    plt.ylabel("V")
    ax1.axhline(reference,color = "b")
    ax1.plot(xxx,data[whichchannel-1][scope_low:scope_high],"r-",label = "channel %d"%(whichchannel))
    plt.legend(loc = 1,shadow = True)
    plt.show()
    
# set label for eeg leftreight(use channel 2 as reference)
def eeg_leftright_set_label(data,                          # numpy array: data
                            reference,                     # numpy array: reference of corresponding data
                            left_threshold = 70,           # float: threshold for moving left
                            right_threshold = -70,         # float: threshold for moving right
                            notmoving_threshold = 10,      # float: threshold for not moving
                            label_dim = 5,                 # int: dimension of one-hot coding
                            notmovingwhere = 0,            # int: index of where notmoving should be in one-hot coding
                            leftwhere = 1,                 # int: index of where moving left should be in one-hot coding
                            rightwhere = 2):               # int: index of where moving right should be in one-hot coding
    
    if (notmoving_threshold > left_threshold) or (notmoving_threshold > -right_threshold):
        return print("notmoving_threshold cannot bigger than absolute value of left_threshold and right_threshold")
    
    reference_value = np.zeros((4,1))
    for i in range(4):             # save the average value for 4 channels
        reference_value[i][0] = np.mean(reference[i])

    outputdata = np.array([]).reshape(4,0)
    label = np.array([]).reshape(0,label_dim)       #one-hot coding label with 5 value [notmoving,left,right,up,down]
    tmp = np.zeros((1,label_dim),dtype = np.float32) 
    for i in range(len(data[1])):   # use channel 2 as indicator
        if data[1][i] >= reference_value[1][0] + left_threshold:
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][leftwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][leftwhere] = 0.0
        elif data[1][i] <= reference_value[1][0] + right_threshold:
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][rightwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][rightwhere] = 0.0
        elif (data[1][i] < reference_value[1][0] + notmoving_threshold) and (data[1][i] > reference_value[1][0] - notmoving_threshold):
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][notmovingwhere] = 0.0
    return outputdata.T, label

# set label for eeg updown(use channel 4 as reference)
def eeg_updown_set_label(data,                                  # numpy array: data                        
                         reference,                             # numpy array: reference of corresponding data
                         up_threshold = -70,                    # float: threshold for moving up
                         down_threshold = 70,                   # float: threshold for moving down
                         notmoving_threshold = 10,              # float: threshold for not moving
                         label_dim = 5,                         # int: dimension of one-hot coding
                         notmovingwhere = 0,                    # int: index of where notmoving should be in one-hot coding
                         upwhere = 3,                           # int: index of where moving up should be in one-hot coding
                         downwhere = 4):                        # int: index of where moving down should be in one-hot coding
    
    if (notmoving_threshold > down_threshold) or (notmoving_threshold > -up_threshold):
        return print("notmoving_threshold cannot bigger than absolute value of up_threshold and down_threshold")

    reference_value = np.zeros((4,1))
    for i in range(4):             # save the average value for 4 channels
        reference_value[i][0] = np.mean(reference[i])
        
    outputdata = np.array([]).reshape(4,0)
    label = np.array([]).reshape(0,label_dim)       #one-hot coding label with 5 value [notmoving,left,right,up,down]
    tmp = np.zeros((1,label_dim),dtype = np.float32) 
    for i in range(len(data[3])):   # use channel 2 as indicator
        if data[3][i] <= reference_value[3][0] + up_threshold:
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][upwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][upwhere] = 0.0
        elif data[3][i] >= reference_value[3][0] + down_threshold:
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][downwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][downwhere] = 0.0
        elif (data[3][i] < reference_value[3][0] + notmoving_threshold) and (data[3][i] > reference_value[3][0] - notmoving_threshold):
            outputdata = np.hstack([outputdata, data[:,i].reshape(4,1)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp]) 
            tmp[0][notmovingwhere] = 0.0  
    return outputdata.T, label

# prepare the training valid and test set for training (randomly split all data to 70% training, 15% valid and 15% test)
# will automaticly build a balance data set (mean it will delete some superfluous data)!!!!
def prepare_dataset(datas,                                   # list of numpy array: data
                    labels,                                  # list of numpy array: corresponding label (must in same order with data)
                    random_seed = 516):                      # int: random seed (so training, valid and test set will be fixed)
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    dataset = np.vstack(datas)
    labelset = np.vstack(labels)
    label_dim = labelset.shape[1]
    input_dim = dataset.shape[1]
    
    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    notmoving_list = []    # used to build balance dataset
    delete_list = []
    for i in range(labelset.shape[0]):
        if labelset[i][0] == 1.0:
            counter_0 += 1
            notmoving_list.append(i)
        if labelset[i][1] == 1.0:
            counter_1 += 1
        if labelset[i][2] == 1.0:
            counter_2 += 1
        if labelset[i][3] == 1.0:
            counter_3 += 1
        if labelset[i][4] == 1.0:
            counter_4 += 1
    if len(notmoving_list) > counter_1 + counter_2 + counter_3 + counter_4:
        random_index = random.sample(range(len(notmoving_list)),int(len(notmoving_list)-(counter_1 + counter_2 +counter_3 + counter_4)/4))
        for i in random_index:
            delete_list.append(notmoving_list[i])
        dataset = np.delete(dataset,delete_list,0)
        labelset = np.delete(labelset,delete_list,0)
    
    random_index = np.random.permutation(np.shape(dataset)[0])
    size_of_training_dataset = int(np.shape(dataset)[0] * 0.7)    # put 70% into training set
    size_of_test_dataset = int(np.shape(dataset)[0] * 0.15)    # put 15% into test set
    size_of_valid_dataset = np.shape(dataset)[0] - size_of_training_dataset - size_of_test_dataset  # put rest into validation set

    # initialize training, validation and test dataset
    training_dataset = np.zeros((size_of_training_dataset, input_dim),dtype = np.float32)
    training_labelset = np.zeros((size_of_training_dataset, label_dim),dtype = np.float32)
    valid_dataset = np.zeros((size_of_valid_dataset, input_dim),dtype = np.float32)
    valid_labelset = np.zeros((size_of_valid_dataset, label_dim),dtype = np.float32)
    test_dataset = np.zeros((size_of_test_dataset, input_dim),dtype = np.float32)
    test_labelset = np.zeros((size_of_test_dataset, label_dim),dtype = np.float32)

    # generate training, validation and test dataset
    for i in range(0,size_of_training_dataset):
        training_dataset[i] = dataset[random_index[i]]
        training_labelset[i] = labelset[random_index[i]]

    for i in range(size_of_training_dataset, size_of_training_dataset + size_of_valid_dataset):
        valid_dataset[i - size_of_training_dataset] = dataset[random_index[i]]
        valid_labelset[i - size_of_training_dataset] = labelset[random_index[i]]

    for i in range(size_of_training_dataset + size_of_valid_dataset,size_of_training_dataset + size_of_valid_dataset + size_of_test_dataset):
        test_dataset[i - (size_of_training_dataset + size_of_valid_dataset)] = dataset[random_index[i]]
        test_labelset[i - (size_of_training_dataset + size_of_valid_dataset)] = labelset[random_index[i]]
    return training_dataset, training_labelset, valid_dataset, valid_labelset, test_dataset, test_labelset

# set up and down label for a window of data
def eeg_extract_window_updown_set_label(data,                                  # numpy array: data 
                                        reference,                             # numpy array: reference of corresponding data
                                        up_threshold = -70,                    # float: threshold for moving up
                                        down_threshold = 70,                   # float: threshold for moving down
                                        notmoving_threshold = 10,              # float: threshold for not moving
                                        window_leight = 10,                    # int: length of window
                                        label_dim = 5,                         # int: dimension of one-hot coding
                                        notmovingwhere = 0,                    # int: index of where notmoving should be in one-hot coding
                                        upwhere = 3,                           # int: index of where moving up should be in one-hot coding
                                        downwhere = 4):                        # int: index of where moving down should be in one-hot coding
    
    if (notmoving_threshold > down_threshold) or (notmoving_threshold > -up_threshold):
        return print("notmoving_threshold cannot bigger than absolute value of up_threshold and down_threshold")

    reference_value = np.zeros((4,1))
    for i in range(4):             # save the average value for 4 channels
        reference_value[i][0] = np.mean(reference[i])
    
    how_many_windows = math.ceil(data.shape[1] / window_leight)
        
    outputdata = np.array([]).reshape(0,4,window_leight)
    label = np.array([]).reshape(0,label_dim)       
    tmp = np.zeros((1,label_dim),dtype = np.float32) 
    for i in range(how_many_windows - 1):   # use channel 2 as indicator
        if np.mean(data[3][i*window_leight:(i+1)*window_leight]) <= reference_value[3] + up_threshold:
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][upwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][upwhere] = 0.0
        elif np.mean(data[3][i*window_leight:(i+1)*window_leight]) >= reference_value[3] + down_threshold:
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][downwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][downwhere] = 0.0            
        elif (np.mean(data[3][i*window_leight:(i+1)*window_leight]) < reference_value[3] + notmoving_threshold)          and (np.mean(data[3][i*window_leight:(i+1)*window_leight]) > reference_value[3] - notmoving_threshold):
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp]) 
            tmp[0][notmovingwhere] = 0.0
    remain = how_many_windows * window_leight - len(data[3])
    data = np.concatenate([data,np.zeros([4,remain])],axis = 1)
    if remain != 0:
        if np.mean(data[3][(how_many_windows-1) * window_leight:len(data[3])]) <= reference_value[3] + up_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[3])].reshape(1,4,window_leight)])
            tmp[0][upwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][upwhere] = 0.0
        elif np.mean(data[3][(how_many_windows-1) * window_leight:len(data[3])]) >= reference_value[3] + down_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[3])].reshape(1,4,window_leight)])
            tmp[0][downwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][downwhere] = 0.0            
        elif np.mean(data[3][(how_many_windows-1) * window_leight:len(data[3])]) < reference_value[3] + notmoving_threshold          and np.mean(data[3][(how_many_windows-1) * window_leight:len(data[3])]) > reference_value[3] - notmoving_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[3])].reshape(1,4,window_leight)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp]) 
            tmp[0][notmovingwhere] = 0.0
    return outputdata, label

# set left and right label for a window of data
def eeg_extract_window_leftright_set_label(data,                            # numpy array: data 
                                           reference,                       # numpy array: reference of corresponding data
                                           left_threshold = -70,            # float: threshold for moving left
                                           right_threshold = 70,            # float: threshold for moving right
                                           notmoving_threshold = 10,        # float: threshold for not moving
                                           window_leight = 10,              # int: length of window
                                           label_dim = 5,                   # int: dimension of one-hot coding
                                           notmovingwhere = 0,              # int: index of where notmoving should be in one-hot coding
                                           leftwhere = 1,                   # int: index of where moving left should be in one-hot coding
                                           rightwhere = 2):                 # int: index of where moving right should be in one-hot coding
    
    if (notmoving_threshold > right_threshold) or (notmoving_threshold > -left_threshold):
        return print("notmoving_threshold cannot bigger than absolute value of left_threshold and right_threshold")

    reference_value = np.zeros((4,1))
    for i in range(4):             # save the average value for 4 channels
        reference_value[i][0] = np.mean(reference[i])
    
    how_many_windows = math.ceil(data.shape[1] / window_leight)
        
    outputdata = np.array([]).reshape(0,4,window_leight)
    label = np.array([]).reshape(0,label_dim)       
    tmp = np.zeros((1,label_dim),dtype = np.float32) 
    for i in range(how_many_windows - 1):   # use channel 2 as indicator
        if np.mean(data[1][i*window_leight:(i+1)*window_leight]) <= reference_value[1] + left_threshold:
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][leftwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][leftwhere] = 0.0
        elif np.mean(data[1][i*window_leight:(i+1)*window_leight]) >= reference_value[1] + right_threshold:
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][rightwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][rightwhere] = 0.0            
        elif (np.mean(data[1][i*window_leight:(i+1)*window_leight]) < reference_value[1] + notmoving_threshold)          and (np.mean(data[1][i*window_leight:(i+1)*window_leight]) > reference_value[1] - notmoving_threshold):
            outputdata = np.concatenate([outputdata, data[:,(i*window_leight):((i+1)*window_leight)].reshape(1,4,window_leight)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp]) 
            tmp[0][notmovingwhere] = 0.0
    remain = how_many_windows * window_leight - len(data[1])
    data = np.concatenate([data,np.zeros([4,remain])],axis = 1)
    if remain != 0:
        if np.mean(data[1][(how_many_windows-1) * window_leight:len(data[1])]) <= reference_value[1] + left_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[1])].reshape(1,4,window_leight)])
            tmp[0][leftwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][leftwhere] = 0.0
        elif np.mean(data[1][(how_many_windows-1) * window_leight:len(data[1])]) >= reference_value[1] + right_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[1])].reshape(1,4,window_leight)])
            tmp[0][rightwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][rightwhere] = 0.0            
        elif np.mean(data[1][(how_many_windows-1) * window_leight:len(data[1])]) < reference_value[1] + notmoving_threshold          and np.mean(data[1][(how_many_windows-1) * window_leight:len(data[1])]) > reference_value[1] - notmoving_threshold:
            outputdata = np.concatenate([outputdata, data[:,((how_many_windows-1) * window_leight):len(data[1])].reshape(1,4,window_leight)])
            tmp[0][notmovingwhere] = 1.0
            label = np.vstack([label, tmp])
            tmp[0][notmovingwhere] = 0.0
    return outputdata, label

# prepare the training valid and test set for training (randomly split all data to 70% training, 15% valid and 15% test)
# will automaticly build a balance data set (mean it will delete some superfluous data)!!!!
def prepare_window_dataset(datas,                                   # list of numpy array: data
                           labels,                                  # list of numpy array: corresponding label (must in same order with data)
                           random_seed = 516):                      # int: random seed (so training, valid and test set will be fixed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    dataset = np.vstack(datas)
    labelset = np.vstack(labels)
    window_leight = dataset.shape[2]
    input_dim = dataset.shape[1]
    label_dim = labelset.shape[1]
    
    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    notmoving_list = []    # used to build balance dataset
    delete_list = []
    for i in range(labelset.shape[0]):
        if labelset[i][0] == 1.0:
            counter_0 += 1
            notmoving_list.append(i)
        if labelset[i][1] == 1.0:
            counter_1 += 1
        if labelset[i][2] == 1.0:
            counter_2 += 1
        if labelset[i][3] == 1.0:
            counter_3 += 1
        if labelset[i][4] == 1.0:
            counter_4 += 1
    if len(notmoving_list) > counter_1 + counter_2 + counter_3 + counter_4:
        random_index = random.sample(range(len(notmoving_list)),int(len(notmoving_list)-(counter_1 + counter_2 +counter_3 + counter_4)/4))
        for i in random_index:
            delete_list.append(notmoving_list[i])
        dataset = np.delete(dataset,delete_list,0)
        labelset = np.delete(labelset,delete_list,0)
    
    random_index = np.random.permutation(np.shape(dataset)[0])
    size_of_training_dataset = int(np.shape(dataset)[0] * 0.7)    # put 70% into training set
    size_of_test_dataset = int(np.shape(dataset)[0] * 0.15)    # put 15% into test set
    size_of_valid_dataset = np.shape(dataset)[0] - size_of_training_dataset - size_of_test_dataset  # put rest into validation set

    # initialize training, validation and test dataset
    training_dataset = np.zeros((size_of_training_dataset, input_dim, window_leight),dtype = np.float32)
    training_labelset = np.zeros((size_of_training_dataset, label_dim),dtype = np.float32)
    valid_dataset = np.zeros((size_of_valid_dataset, input_dim, window_leight),dtype = np.float32)
    valid_labelset = np.zeros((size_of_valid_dataset, label_dim),dtype = np.float32)
    test_dataset = np.zeros((size_of_test_dataset, input_dim, window_leight),dtype = np.float32)
    test_labelset = np.zeros((size_of_test_dataset, label_dim),dtype = np.float32)

    # generate training, validation and test dataset
    for i in range(0,size_of_training_dataset):
        training_dataset[i] = dataset[random_index[i]]
        training_labelset[i] = labelset[random_index[i]]

    for i in range(size_of_training_dataset, size_of_training_dataset + size_of_valid_dataset):
        valid_dataset[i - size_of_training_dataset] = dataset[random_index[i]]
        valid_labelset[i - size_of_training_dataset] = labelset[random_index[i]]

    for i in range(size_of_training_dataset + size_of_valid_dataset,size_of_training_dataset + size_of_valid_dataset + size_of_test_dataset):
        test_dataset[i - (size_of_training_dataset + size_of_valid_dataset)] = dataset[random_index[i]]
        test_labelset[i - (size_of_training_dataset + size_of_valid_dataset)] = labelset[random_index[i]]
    return training_dataset, training_labelset, valid_dataset, valid_labelset, test_dataset, test_labelset

# use tenforflow to train a logistic_regression_model
def logistic_regression_model_training(train_dataset,                  
                                       train_labelset,
                                       valid_dataset,
                                       valid_labelset,
                                       test_dataset,
                                       test_labelset,
                                       lr = 0.0002,                   # float: learning rate
                                       epoch = 200,                   # int: max epoch
                                       size_of_batch = 128,           # int: size of each minibatch
                                       weight_decay = 0.0005):        # float: weight dacay for loss function

    size_of_training_dataset = np.shape(train_dataset)[0]
    number_of_batchs = math.ceil(size_of_training_dataset / size_of_batch)

    # use to draw the learning curve
    loss_train = []
    loss_valid = []
    loss_test = []
    acc_train = []
    acc_valid = []
    acc_test = []
    weight_final = 0
    bias_final = 0
    # final accuracy on test set
    acc_test_final = 0

    x = tf.placeholder(tf.float32, [None, train_dataset.shape[1]]) #input
    y = tf.placeholder(tf.float32, [None, train_labelset.shape[1]]) #label

    W = tf.Variable(tf.zeros((train_dataset.shape[1],train_labelset.shape[1])))  #weight
    b = tf.Variable(tf.zeros((train_labelset.shape[1])))  # bias

    # model
    y_hat = tf.add(tf.matmul(x, W),b)

    # cost function sigmoid mean with L2 trade off
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat,labels = y))            + 0.5 * weight_decay * tf.reduce_mean(tf.matmul(tf.matrix_transpose(W),W))

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss) 

    #accuracy
    correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize
    init = tf.global_variables_initializer()

    # parpare minibatch
    input_data,input_label = tf.train.shuffle_batch([train_dataset,train_labelset],
                                                    batch_size = size_of_batch,
                                                    capacity = 50000,
                                                    min_after_dequeue = 10000,
                                                    enqueue_many = True,
                                                    allow_smaller_final_batch = True)
    # training
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        for i in range(epoch):
            for j in range(number_of_batchs):
                minibatch_x,minibatch_y = sess.run([input_data,input_label])
                sess.run(optimizer,feed_dict={x: minibatch_x, 
                                              y: minibatch_y})
            loss_train_tmp,acc_train_tmp = sess.run([loss,accuracy],feed_dict={x: train_dataset, 
                                                                               y: train_labelset})
            loss_valid_tmp,acc_valid_tmp = sess.run([loss,accuracy],feed_dict={x: valid_dataset, 
                                                                               y: valid_labelset})
            loss_test_tmp,acc_test_tmp = sess.run([loss,accuracy],feed_dict={x: test_dataset, 
                                                                             y: test_labelset})
            loss_train.append(loss_train_tmp)
            loss_valid.append(loss_valid_tmp)
            loss_test.append(loss_test_tmp)
            acc_train.append(acc_train_tmp)
            acc_valid.append(acc_valid_tmp)
            acc_test.append(acc_test_tmp)
            print(i)

        acc_test_final = sess.run(accuracy,feed_dict={x: test_dataset, 
                                                      y: test_labelset})
        coord.request_stop()
        coord.join(threads)
        weight_final = W.eval()
        bias_final = b.eval()
    print(weight_final)
    print(bias_final)
    
    # plot
    xxx = np.linspace(1,epoch,epoch,dtype = np.int16)
    
    plt.figure(figsize = (15,10),dpi = 300)
    ax1 = plt.subplot()
    plt.title("Cross Entrpy Loss on Logistic Regression Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax1.plot(xxx,loss_train,'r-',label = "Training")
    ax1.plot(xxx,loss_valid,'b-',label = "Validation")
    plt.legend(loc=1,shadow=True)
    plt.savefig("G:\ECE516\FUI\Logisticloss.jpg")
    plt.show()
    
    plt.figure(figsize = (15,10),dpi = 300)
    ax2 = plt.subplot()
    plt.title("Training performance Logistic Regression Model")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    ax2.plot(xxx,acc_train,'r-',label = "Training Data")
    ax2.plot(xxx,acc_valid,'b-',label = "Validation Data")
    plt.legend(loc=4,shadow=True)
    plt.savefig("G:\\ECE516\\FUI\\Logisticperformance.jpg")
    plt.show()
    print("final_test:",acc_test_final)
    print("max_valid:",max(acc_valid))
    return weight_final, bias_final
    
# use pytorch to train a single hidden layer neural network
def torch_neural_network_training(train_dataset,
                                  train_labelset,
                                  valid_dataset,
                                  valid_labelset,
                                  test_dataset,
                                  test_labelset,
                                  hide = 200,                       # int: number of nodes in hidden layer
                                  Adam = False,                     # Bool: False to use SGD optimizer True to use a Adam optimizer
                                  lr = 0.0002,                      # float: learning rate
                                  momentum = 0.9,                   # float: momentum value used in SGD
                                  epoch = 200,                      # int: max epoch
                                  size_of_batch = 128,              # int: size of each minibatch
                                  random_size = 516,                # int: random seed (so minibatch will be fixed)
                                  weight_decay = 0.0005):           # float: weight dacay for loss function
    
    # prepare minibatch
    torch.manual_seed(516) # random seed
    input_dataset = data.TensorDataset(data_tensor = torch.from_numpy(train_dataset),                                        target_tensor = torch.from_numpy(np.argmax(train_labelset, 1))) 
    loader = data.DataLoader( dataset = input_dataset, batch_size = size_of_batch, shuffle = True)
    
    # nn model
    training_learning_curve = []
    test_learning_curve = []
    valid_learning_curve = []
    loss_lc = []

    xlc_training = Variable(torch.from_numpy(train_dataset), requires_grad=False).type(torch.FloatTensor).cuda()
    xlc_test = Variable(torch.from_numpy(test_dataset), requires_grad=False).type(torch.FloatTensor).cuda()
    xlc_valid = Variable(torch.from_numpy(valid_dataset), requires_grad=False).type(torch.FloatTensor).cuda()


    #weight intialization
    #W = torch.from_numpy(np.random.randn(hide,resolution)/sqrt(resolution)).type(dtype_float)
    def weights_init(m):
        if type(m) == torch.nn.Linear:
            m.bias.data.fill_(1.0)
            m.weight.data.normal_(0,0.02)

    # define model
    model = torch.nn.Sequential(
        torch.nn.Linear(train_dataset.shape[1], hide),
        torch.nn.ReLU(),
        torch.nn.Linear(hide, train_labelset.shape[1]),
    ).cuda()

    model.apply(weights_init)

    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    if Adam == True:
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = lr, 
                                     weight_decay = weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = lr, 
                                    momentum = momentum)

    
    soft_max = torch.nn.Softmax(dim = 1)

    # training
    for times in range(epoch):  
        for i, (batch_x, batch_y) in enumerate(loader):  
            X = Variable(batch_x).type(torch.FloatTensor).cuda()
            Y = Variable(batch_y).type(torch.LongTensor).cuda()

            y_pred = model(X)
            loss = loss_fn(y_pred, Y)

            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to 
                               # make a step

        loss_lc.append(loss.data[0])
        # collect data for learning curce (training)
        ylc_training = soft_max(model(xlc_training)).cpu().data.numpy()
        counter = 0
        for i in range(np.shape(ylc_training)[0]):
            k = np.argmax(ylc_training[i])
            for j in range(np.shape(ylc_training)[1]):
                if j == k:
                    ylc_training[i][j] = 1
                else:
                    ylc_training[i][j] = 0
            if np.equal(ylc_training[i],train_labelset[i]).all():
                counter += 1
        training_learning_curve.append(counter/np.shape(train_labelset)[0])


        # collect data for learning curce (test)
        ylc_test = soft_max(model(xlc_test)).cpu().data.numpy()
        counter = 0
        for i in range(np.shape(ylc_test)[0]):
            k = np.argmax(ylc_test[i])
            for j in range(np.shape(ylc_test)[1]):
                if j == k:
                    ylc_test[i][j] = 1
                else:
                    ylc_test[i][j] = 0
            if np.equal(ylc_test[i],test_labelset[i]).all():
                counter += 1
        test_learning_curve.append(counter/np.shape(test_labelset)[0])

        # collect data for learning curce (validation)
        ylc_valid = soft_max(model(xlc_valid)).cpu().data.numpy()
        counter = 0
        for i in range(np.shape(ylc_valid)[0]):
            k = np.argmax(ylc_valid[i])
            for j in range(np.shape(ylc_valid)[1]):
                if j == k:
                    ylc_valid[i][j] = 1
                else:
                    ylc_valid[i][j] = 0
            if np.equal(ylc_valid[i],valid_labelset[i]).all():
                counter += 1
        valid_learning_curve.append(counter/np.shape(valid_labelset)[0])
        print(times)
    

    # final test performance
    ylc_test = soft_max(model(xlc_test)).cpu().data.numpy()
    final_counter = 0
    for i in range(np.shape(ylc_test)[0]):
        k = np.argmax(ylc_test[i])
        for j in range(np.shape(ylc_test)[1]):
            if j == k:
                ylc_test[i][j] = 1
            else:
                ylc_test[i][j] = 0
        if np.equal(ylc_test[i],test_labelset[i]).all():
            final_counter += 1
        final_acc = final_counter/np.shape(test_labelset)[0]
        
    params = model.state_dict()
    final_weight_0 = params['0.weight'] 
    final_bias_0 = params['0.bias']
    final_weight_2 = params['2.weight']
    final_bias_2 = params['2.bias'] 

    # plot
    xxx = range(epoch)
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(xxx,loss_lc,'y-')
    plt.show()
        
    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.1,1.1)
    plt.xlim(0,epoch+20)
    l1, = plt.plot(xxx,training_learning_curve,'b')
    l2, = plt.plot(xxx,valid_learning_curve,'r-')
    l3, = plt.plot(xxx,test_learning_curve,'g-')
    plt.legend(handles = [l1, l2, l3], labels = ['Training Set', 'Validation Set','Test Set'], loc = 'lower right')
    plt.show()
    
    print(final_acc)
    return [final_weight_0,final_weight_2 ], [final_bias_0, final_bias_2]

# use tensorflow to build a 1D convolution nerual network
def OneD_CNN_model_training(train_dataset,  
                            train_labelset,
                            valid_dataset,
                            valid_labelset,
                            test_dataset,
                            test_labelset,
                            lr = 0.0002,              # float: learning rate
                            epoch = 200,              # int: max epoch
                            size_of_batch = 128):     # int: size of each minibatch

    size_of_training_dataset = train_dataset.shape[0]
    number_of_batchs = math.ceil(size_of_training_dataset / size_of_batch)
    
    # use to draw the learning curve
    loss_train = []
    loss_valid = []
    loss_test = []
    acc_train = []
    acc_valid = []
    acc_test = []
    weight_final = 0
    bias_final = 0
    # final accuracy on test set
    acc_test_final = 0

    x = tf.placeholder(tf.float32, [None, train_dataset.shape[1], train_dataset.shape[2]]) #input
    y = tf.placeholder(tf.float32, [None, train_labelset.shape[1]]) #label

    #W = tf.Variable(tf.zeros((train_dataset.shape[1],train_labelset.shape[1])))  #weight
    #b = tf.Variable(tf.zeros((train_labelset.shape[1])))  # bias

    # model
    # first conv
    conv1 = tf.layers.conv1d(inputs = x,
                             filters = 4,
                             kernel_size = 4,
                             strides = 1,
                             padding='same',
                             data_format='channels_first',
                             activation=tf.nn.relu)
    # first pooling
    #pool1 = tf.layers.max_pooling1d(inputs = conv1,pool_size = 2,strides = 2,padding='valid',data_format='channels_first')
            
    # second conv
    conv2 = tf.layers.conv1d(inputs = conv1,
                             filters = 4,
                             kernel_size = 2,
                             strides = 2,
                             padding='same',
                             data_format='channels_first',
                             activation=tf.nn.relu)
    conv2_flat = tf.reshape(conv2, [-1, 4 * 5])

    # dense layer
    dense1 = tf.layers.dense(inputs = conv2_flat, units=10, activation=tf.nn.relu)
    
    # output layer
    y_hat = tf.layers.dense(inputs=dense1, units=5)
    
    # cost function sigmoid mean with L2 trade off
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat,labels = y))
           # + 0.5 * weight_decay * tf.reduce_mean(tf.matmul(tf.matrix_transpose(W),W))   

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    #accuracy
    correct_pred = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize
    init = tf.global_variables_initializer()

    # parpare minibatch
    input_data,input_label = tf.train.shuffle_batch([train_dataset,train_labelset],
                                                    batch_size = size_of_batch,
                                                    capacity = 50000,
                                                    min_after_dequeue = 10000,
                                                    enqueue_many = True,
                                                    allow_smaller_final_batch = True)
    # training
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        for i in range(epoch):
            for j in range(number_of_batchs):
                minibatch_x,minibatch_y = sess.run([input_data,input_label])
                sess.run(optimizer,feed_dict={x: minibatch_x, 
                                              y: minibatch_y})
            loss_train_tmp,acc_train_tmp = sess.run([loss,accuracy],feed_dict={x: train_dataset, 
                                                                               y: train_labelset})
            loss_valid_tmp,acc_valid_tmp = sess.run([loss,accuracy],feed_dict={x: valid_dataset, 
                                                                               y: valid_labelset})
            loss_test_tmp,acc_test_tmp = sess.run([loss,accuracy],feed_dict={x: test_dataset, 
                                                                             y: test_labelset})
            loss_train.append(loss_train_tmp)
            loss_valid.append(loss_valid_tmp)
            loss_test.append(loss_test_tmp)
            acc_train.append(acc_train_tmp)
            acc_valid.append(acc_valid_tmp)
            acc_test.append(acc_test_tmp)
            print(i)

        acc_test_final = sess.run(accuracy,feed_dict={x: test_dataset, 
                                                      y: test_labelset})
        coord.request_stop()
        coord.join(threads)
        #weight_final = W.eval()
        #bias_final = b.eval()

    
    # plot
    xxx = np.linspace(1,epoch,epoch,dtype = np.int16)
    
    plt.figure(figsize = (15,10),dpi = 300)
    ax1 = plt.subplot()
    plt.title("Cross Entrpy Loss",fontsize = 26)
    plt.xlabel("Epoch",fontsize = 26)
    plt.ylabel("Loss",fontsize = 26)
    ax1.plot(xxx,loss_train,'r-',label = "Training Data")
    ax1.plot(xxx,loss_valid,'b-',label = "Validatation Data")
    plt.legend(loc=1,shadow=True, prop = {"size" : 22})
    plt.savefig("G:\ECE516\FUI\\CNN_loss2.jpg")
    plt.show()
    
    plt.figure(figsize = (15,10),dpi = 300)
    ax2 = plt.subplot()
    plt.title("Training Performance using CNN",fontsize = 26)
    plt.xlabel("Epoch",fontsize = 26)
    plt.ylabel("Accuracy",fontsize = 26)
    ax2.plot(xxx,acc_train,'r-',label = "Training Data")
    ax2.plot(xxx,acc_valid,'b-',label = "Validation Data")
    plt.legend(loc=4,shadow=True,prop = {"size" : 22})
    plt.savefig("G:\ECE516\FUI\\CNN_performance2.jpg")
    plt.show()
    print("max vlaid:",max(acc_valid))
    print("max vlaid index:",acc_valid.index(max(acc_valid)))
    print("final test:",acc_test_final)
    
# make a 3D plot of eeg channel
def plot_eeg_3D(data,                        # numpy array: data to plot
                scope_low,                   # int: lower boundary of data to plot 
                scope_high,                  # int: upper boundary of data to plot 
                x_channel = 2,               # int: which channel should be put at X axis  (Y axis represent time)
                z_channel = 4,               # int: which channel should be put at Z axis
                mode = 1):                   # int(from 1-5): which style should be use
    if scope_high > len(data[0]):
        print("higher boundary out of limits")
        return
    color = [0.0, 0.0, 0.0]
    xxx = data[x_channel-1][scope_low:scope_high]                   
    yyy = range(0,len(data[0]))[scope_low:scope_high]                # time
    zzz = data[z_channel-1][scope_low:scope_high]                    
    
    x_reference = [np.mean(xxx)]*len(xxx)
    z_reference = [np.mean(zzz)]*len(zzz)
    x_reference_value = np.mean(xxx)
    z_reference_value = np.mean(zzz)
    radius = []
    color_tmp = []
    fig = plt.figure(figsize = (20,7))
    ax = plt.subplot(111,projection = "3d")
    ax.view_init(elev=10., azim=11)
    #plt.title("Data")
    ax.set_xlabel('Channel %d' % x_channel)
    ax.set_ylabel('Time')
    ax.set_zlabel('Channel %d' % z_channel)
    if mode == 1:
        ax.plot3D(xxx,yyy,x_reference,"b-",label = "Channel %d" % x_channel)
        ax.plot3D(z_reference,yyy,zzz,"r-",label = "Channel %d" % z_channel)
        plt.legend(loc = 1,shadow = True)
        plt.show()
    elif mode == 2:
        ax.plot3D(xxx,yyy,zzz,"b-",label = "Time Line")
        plt.legend(loc = 1,shadow = True)
        plt.show()
    elif mode == 3:
        ax.plot3D(xxx,yyy,zzz,"b.",label = "Time Line")
        plt.legend(loc = 1,shadow = True)
        plt.show()
    elif mode == 4:
        for i in range(len(yyy)):
            radius.append(math.sqrt((xxx[i] - x_reference_value)**2 + (zzz[i] - z_reference_value)**2))
        radius_max = max(radius)
        ax.set_xlim(x_reference_value - radius_max - 20,x_reference_value + radius_max + 20)
        ax.set_ylim(scope_low,scope_high)
        ax.set_zlim(z_reference_value - radius_max - 20,z_reference_value + radius_max + 20)
        for i in range(len(yyy)):
            tmp = radius[i]/radius_max
            p = Circle((x_reference_value, z_reference_value), radius[i], fill = False, color = (tmp,1-tmp,1))
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=yyy[i], zdir="y")
        plt.legend(loc = 1,shadow = True)
        plt.show()
    elif mode == 5:
        for i in range(len(yyy)):
            radius.append(math.sqrt((xxx[i] - x_reference_value)**2 + (zzz[i] - z_reference_value)**2))
        radius_max = max(radius)
        for i in range(len(yyy)):
            color_tmp.append(1 - radius[i]/radius_max)
        sc = ax.scatter(xxx,yyy,zzz,cmap = "cool",marker = "." ,c = color_tmp)
        plt.colorbar(sc,shrink = 0.7, orientation = "horizontal", aspect = 50)
        plt.legend(loc = 1,shadow = True)
        plt.show()
    else:
        return print("don't have this mode!")


# In[99]:


# example of importing data
reference_zhangge = eeg_extract("G:\\ECE516\\FUI\\data\\reference_zhangge.txt")
leftright_zhangge = eeg_extract("G:\\ECE516\\FUI\\data\\eyeballmoving_zhangge.txt")
updown_zhangge = eeg_extract("G:\\ECE516\\FUI\\data\\eyeball_uptown_zhangge.txt")

reference_zw = eeg_extract("G:\\ECE516\\FUI\\data\\reference_zw.txt")
leftright_zw = eeg_extract("G:\\ECE516\\FUI\\data\\eyeball_leftright_zw.txt")
updown_zw = eeg_extract("G:\\ECE516\\FUI\\data\\eyeball_updown_zw.txt")

reference_yiqun = eeg_extract("G:\\ECE516\\FUI\\data\\reference_yiqun.txt")
leftright_yiqun = eeg_extract("G:\\ECE516\\FUI\\data\\eyeball_leftright_yiqun.txt")
updown_yiqun = eeg_extract("G:\\ECE516\\FUI\\data\\eyeball_updown_yiyun.txt")

sleeping_zw = eeg_extract("G:\\ECE516\\FUI\\paper\\sleeping_zw.txt")
reference_zw = eeg_extract("G:\\ECE516\\FUI\\data\\reference_zw.txt")


# In[24]:


# example of building datasets
dataset_updown_zhangge,label_updown_zhangge = eeg_updown_set_label(updown_zhangge,reference_zhangge,
                                                                   up_threshold = -70, 
                                                                    down_threshold = 70, 
                                                                    notmoving_threshold = 70)

dataset_updown_zw,label_updown_zw = eeg_updown_set_label(updown_zw,reference_zw,
                                                         up_threshold = -70, 
                                                         down_threshold = 70, 
                                                         notmoving_threshold = 70)

dataset_updown_yiqun,label_updown_yiqun = eeg_updown_set_label(updown_yiqun,reference_yiqun,
                                                               up_threshold = -70, 
                                                               down_threshold = 70, 
                                                               notmoving_threshold = 70)


dataset_leftright_zhangge,label_leftright_zhangge = eeg_leftright_set_label(leftright_zhangge,reference_zhangge,
                                                                            left_threshold = 60, 
                                                                            right_threshold = -60,
                                                                            notmoving_threshold = 60)

dataset_leftright_zw,label_leftright_zw = eeg_leftright_set_label(leftright_zw,reference_zw,
                                                                  left_threshold = 60, 
                                                                  right_threshold = -60,
                                                                  notmoving_threshold = 60)

dataset_leftright_yiqun,label_leftright_yiqun = eeg_leftright_set_label(leftright_yiqun,reference_yiqun,
                                                                        left_threshold = 60, 
                                                                        right_threshold = -60,
                                                                        notmoving_threshold = 60)


train_x,train_y,valid_x,valid_y,test_x,test_y = prepare_dataset([dataset_updown_zhangge,
                 dataset_updown_zw,
                 dataset_updown_yiqun,
                 dataset_leftright_zhangge,
                 dataset_leftright_zw,
                 dataset_leftright_yiqun
                ],
                [label_updown_zhangge,
                 label_updown_zw,
                 label_updown_yiqun,
                 label_leftright_zhangge,
                 label_leftright_zw,
                 label_leftright_yiqun
                ],
                random_seed = 500)


# In[26]:


# example of trianing a logistic regression model
weight,bias = logistic_regression_model_training(train_x,
                                   train_y,
                                   valid_x,
                                   valid_y,
                                   test_x,
                                   test_y,
                                   lr = 0.00002,
                                   epoch = 2000,
                                   size_of_batch = 256,
                                   weight_decay = 0.0005)


# In[ ]:


# example of trianing a pytorch nerual network
weight,bias = torch_neural_network_training(train_x,
                              train_y,
                              valid_x,
                              valid_y,
                              test_x,
                              test_y,
                              hide = 10,
                              Adam = True,
                              lr = 0.00008,
                              momentum = 0.9,
                              epoch = 300,
                              size_of_batch = 256,
                              random_size = 516,
                              weight_decay = 0.0005)


# In[5]:


# example of preparing a convolution nerual network dataset
win_dataset_updown_zhangge,win_label_updown_zhangge = eeg_extract_window_updown_set_label(updown_zhangge,reference_zhangge,
                                                                                          up_threshold = -70, 
                                                                                          down_threshold = 70, 
                                                                                          notmoving_threshold = 70,
                                                                                          window_leight = 10)

win_dataset_updown_zw,win_label_updown_zw = eeg_extract_window_updown_set_label(updown_zw,reference_zw,
                                                                                          up_threshold = -70, 
                                                                                          down_threshold = 70, 
                                                                                          notmoving_threshold = 70,
                                                                                          window_leight = 10)

win_dataset_updown_yiqun,win_label_updown_yiqun = eeg_extract_window_updown_set_label(updown_yiqun,reference_yiqun,
                                                                                          up_threshold = -70, 
                                                                                          down_threshold = 70, 
                                                                                          notmoving_threshold = 70,
                                                                                          window_leight = 10)



win_dataset_leftright_zhangge,win_label_leftright_zhangge = eeg_extract_window_leftright_set_label(leftright_zhangge,reference_zhangge,
                                                                                          left_threshold = -60, 
                                                                                          right_threshold = 60, 
                                                                                          notmoving_threshold = 60,
                                                                                          window_leight = 10)

win_dataset_leftright_zw,win_label_leftright_zw = eeg_extract_window_leftright_set_label(leftright_zw,reference_zw,
                                                                                          left_threshold = -60, 
                                                                                          right_threshold = 60, 
                                                                                          notmoving_threshold = 60,
                                                                                          window_leight = 10)

win_dataset_leftright_yiqun,win_label_leftright_yiqun = eeg_extract_window_leftright_set_label(leftright_yiqun,reference_yiqun,
                                                                                          left_threshold = -60, 
                                                                                          right_threshold = 60, 
                                                                                          notmoving_threshold = 60,
                                                                                          window_leight = 10)

train_x,train_y,valid_x,valid_y,test_x,test_y = prepare_window_dataset([win_dataset_updown_zhangge,
                        win_dataset_updown_zw,
                        win_dataset_updown_yiqun,
                        win_dataset_leftright_zhangge,
                        win_dataset_leftright_zw,
                        win_dataset_leftright_yiqun
                       ],
                       [win_label_updown_zhangge,
                        win_label_updown_zw,
                        win_label_updown_yiqun,
                        win_label_leftright_zhangge,
                        win_label_leftright_zw,
                        win_label_leftright_yiqun
                       ])
print(train_x.shape)
print(train_y.shape)


# In[14]:


# example of training a 1D convoluation nerual network 
OneD_CNN_model_training(train_x,
                           train_y,
                           valid_x,
                           valid_y,
                           test_x,
                           test_y,
                           lr = 0.0002,
                           epoch = 12000,
                           size_of_batch = 64)


# In[124]:


# example of make a 3D plot
plot_eeg_3D(sleeping_zw, 958000, 959000,mode = 1, z_channel = 4, x_channel = 2)