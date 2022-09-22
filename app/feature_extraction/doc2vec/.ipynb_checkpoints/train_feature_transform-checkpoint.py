import os
import sys
import pandas as pd
from sklearn import preprocessing
import numpy as np

vector_features_dir = "/projets/depression/eRisk_vector_data"
year = "2017"
features_dir = "/projets/sig/mullah/nlp/depression/features"

#preparing training dataset (all together into 1 chunk)
train_vector_input_features_dir = os.path.join(vector_features_dir, 'depression_'+str(year), 'train')
train_vector_output_features_dir = os.path.join(features_dir, year, 'train', 'doc2vec')

labels = ['p', 'n'] #positive and negative labels
users_id = []
features = []
labels_id = []
            
#negative cases
neg_train_vector_input_features_dir = os.path.join(train_vector_input_features_dir, 'neg')
for root, subdirs, files in os.walk(neg_train_vector_input_features_dir):
    for filename in files:        
        labels_id.append(labels[1]) #negative label

        user_id = filename[:-4] #whole filename without extension
        if user_id not in users_id:
            users_id.append(user_id)

        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as fread:
            line = fread.readline()
            features.append(line.rstrip().split(','))

#positive cases
pos_train_vector_input_features_dir = os.path.join(train_vector_input_features_dir, 'pos')
for root, subdirs, files in os.walk(pos_train_vector_input_features_dir):
    for filename in files:
        labels_id.append(labels[0]) #positive label

        user_id = filename[:-4] #whole filename without extension
        if user_id not in users_id:
            users_id.append(user_id)
        else:
            print (filename)

        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as fread:
            line = fread.readline()
            features.append(line.rstrip().split(','))

print ("Total users: {}".format(len(users_id)))
print ("Total features: {}".format(len(features)))
print ("Total labels: {}".format(len(labels_id)))

total_features = len(features[0])
features_id = ['d'+str(i) for i in range(1, total_features+1)]
cols = ['uid'] + features_id + ['class']

features_np = np.asarray(features)
print (features_np.shape)

data = [users_id]
for idx in range(0, len(features[0])):
    feature = features_np[:, idx]
    data.append(list(feature))
data.append(labels_id)

data_np = np.asarray(data)
data_np_t = data_np.T
data_t = data_np_t.tolist()
dataset = pd.DataFrame(data_t, columns=cols)
print (dataset.head())

train_features_path = os.path.join(train_vector_output_features_dir, 'train_eRisk_depression.csv')
print (train_features_path)
dataset.to_csv(train_features_path, index=False)

