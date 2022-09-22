import os
import sys
import pandas as pd
from sklearn import preprocessing
import numpy as np

vector_features_dir = "/projets/depression/eRisk_vector_data"
year = "2018"
features_dir = "/projets/sig/mullah/nlp/depression/features"

#preparing training dataset (all together into 1 chunk)
test_vector_input_features_dir = os.path.join(vector_features_dir, 'depression_'+str(year), 'test')
test_vector_output_features_dir = os.path.join(features_dir, year, 'test', 'doc2vec')

labels = ['p', 'n'] #positive and negative labels

for chunk in range(1, 11):
    print ("Chunk: {}".format(chunk))
    users_id = []
    features = []
    labels_id = []
    for root, subdirs, files in os.walk(test_vector_input_features_dir):
        for filename in files:        
            labels_id.append(labels[1]) #negative label
        
            user_id = filename[:-4]
            if user_id not in users_id:
                users_id.append(user_id)
        
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as fread:
                lines = fread.readlines()
                line = lines[chunk-1]
                features.append(line.rstrip().split(','))

    print ("Total users: {}".format(len(users_id)))
    print ("Total users: ({}:{})".format(len(features), len(features[0])))
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

    test_features_path = os.path.join(test_vector_output_features_dir, 'test_eRisk_depression_'+str(chunk)+'.csv')
    print (test_features_path)
    dataset.to_csv(test_features_path, index=False)

