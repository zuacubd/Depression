import sys
import os
import collections
import csv
import argparse
import random
import re
import emoji
import pickle

import xml.etree.ElementTree as et
import numpy as np
import pandas as pd

import tensorflow as tf
import torch
import torch.nn as nn
import transformers

from datetime import datetime
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
#from load import *

from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class BERT_Arch(nn.Module):
	'''
		defining the architecture
	'''
	def __init__(self, bert):
		super(BERT_Arch, self).__init__()
		self.bert = bert
		# dropout layer
		self.dropout = nn.Dropout(0.1)
                # relu activation function
		self.relu =  nn.ReLU()
		# dense layer 1
		self.fc1 = nn.Linear(768,512)

		# dense layer 2 (Output layer)
		self.fc2 = nn.Linear(512,2)
		#softmax activation function
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, sent_id, mask):
		'''
			define the forward pass
		'''
		#pass the inputs to the model  
		_, cls_hs = self.bert(sent_id, attention_mask=mask)
		x = self.fc1(cls_hs)
		x = self.relu(x)
		x = self.dropout(x)

		# output layer
		x = self.fc2(x)
		# apply softmax activation
		x = self.softmax(x)
		return x

def get_bert_tokenizer(model_name):
	#import bert pre-trained model
	bert = AutoModel.from_pretrained(model_name)
	#load the bert tokenizer
	tokenizer = BertTokenizerFast.from_pretrained(model_name)

	return bert, tokenizer


def freeze_bert_parameters():
	'''
		freeze all the parameters
	'''
	for param in bert.parameters():
		param.requires_grad = False


def initialize_model(bert_base):
	'''
		# pass the pre-trained BERT to our define architecture
	'''
	model = BERT_Arch(bert_base)
	return model


def load_trained_model(model, model_path):
        '''
            load weights of best model
        '''
        model.load_state_dict(torch.load(model_path))


def get_tokenize_encoded_sequence(data_df, max_len, padding=True, truncation=True):
	'''
		tokenize and encode sequences in the training set
	'''
	tokens_encoded = tokenizer.batch_encode_plus(
		data_df.tolist(),
		max_length = max_len,
		padding=padding,
		truncation=truncation
	)
	return tokens_encoded


def get_list_to_tensors(data_list):
	'''
		convert lists to tensors
	'''
	data_seq = torch.tensor(data_list)
	return data_seq


def get_list_to_tensors_float(data_list):
	'''
		converting list of class weights to a tensor
	'''
	data_tensor= torch.tensor(data_list, dtype=torch.float)
	return data_tensor


def get_tensor_dataset(seq, mask, y):
	# wrap tensors
	tensor_data = TensorDataset(seq, mask, y)
	return tensor_data


def get_sequential_sampler(data):
	'''
		sequential sampler for sampling the data during evaluation
	'''
	data_sampler = SequentialSampler(data)
	return data_sampler


def get_data_loader(data, data_sampler, bs):
	'''
		dataLoader for train set
	'''
	dataloader = DataLoader(data, sampler=data_sampler, batch_size=bs)
	return dataloader


def load_data(data_path):
	'''
		loading data with pandas
	'''
	df = pd.read_csv(data_path)
	return df


if __name__ == '__main__':
        print ("Starting ...")
        device = torch.device("cuda") # push the model to GPU
        bert, tokenizer = get_bert_tokenizer('bert-base-uncased')
        freeze_bert_parameters() #freezing all the parameters in pre-trained model
        model = initialize_model(bert)
        model = model.to(device) # push the model to GPU

        #Setting the datapath
        year = int(sys.argv[1])
        cumulative = 0
        ml = "bert"
        pretrained = "base_uncased"
        dist = "balanced" #"natural"
        aggregation = "mean" #max
        model_name = "saved_weights_"+dist+".pt"
        model_state = "finetuned" #pretrained
        max_len = 128
        batch_size = 32

        root_dir = "/projets/sig/mullah/nlp/depression"
        processed_test_dir = os.path.join(root_dir, "data", "processed", str(year), "test")
        if cumulative == 1:
            processed_test_dir = os.path.join(processed_test_dir, "cumulative")

        models_dir = os.path.join(root_dir, "models", str(year), ml, pretrained, model_state)

        prediction_test_dir=os.path.join(root_dir,"prediction",str(year),ml,"test",pretrained,model_state,dist,aggregation)
        if cumulative == 1:
            prediction_test_dir = os.path.join(prediction_test_dir, "cumulative")

        if not os.path.exists(prediction_test_dir):
            os.makedirs(prediction_test_dir)

        model_path = os.path.join(models_dir, model_name)
        load_trained_model(model, model_path)
        model.eval() #so dropout is stopped during prediciton

        chunks = [i for i in range(1, 11)]
        for chunk_num in chunks:
            print ("Predicting chunk:{}".format(chunk_num))
            processed_chunk_test_dir = os.path.join(processed_test_dir, "chunk "+str(chunk_num))
            users_id = []
            preds_score = []
            for root, subdirs, files in os.walk(processed_chunk_test_dir):
                for filename in files:
                    user_id = filename[:-4] #whole filename without extension

                    #if user_id not in users_id:
                    users_id.append(user_id)

                    file_path = os.path.join(root, filename)
                    print ("Processing file: {}".format(filename))
                    with open(file_path, 'r') as fread:
                        data_df = load_data(file_path)
                        data_df = data_df.dropna()

                        test_text = data_df["texts"]
                        test_labels = data_df["labels"]
                        if len(test_text) < 1:
                            preds_score.append(0.0)
                            continue

                        tokens_test = get_tokenize_encoded_sequence(test_text, max_len, True, True)
                        test_seq = get_list_to_tensors(tokens_test['input_ids'])
                        test_mask = get_list_to_tensors(tokens_test['attention_mask'])
                        test_y = get_list_to_tensors(test_labels.tolist())

                        test_data = get_tensor_dataset(test_seq, test_mask, test_y) # wrap tensors
                        test_sampler = get_sequential_sampler(test_data)
                        test_dataloader = get_data_loader(test_data, test_sampler, batch_size)# dataLoader for testset

                        predictions = []
                        for batch in test_dataloader:

                            #batch = tuple(t.to(device) for t in batch)
                            b_test_seq, b_test_mask, b_test_y = batch

                            # get predictions for test data
                            with torch.no_grad():
                                preds = model(b_test_seq.to(device), b_test_mask.to(device))
                                pred_prob = torch.exp(preds) #to_probability
                                pred_preds_np = pred_prob.detach().cpu().numpy()
                                pred_cls = pred_preds_np[:,1]
                                predictions.extend(list(pred_cls))

                        if aggregation == "mean":
                            pred_score = np.mean(predictions)
                        else:
                            pred_score = np.max(predictions)
                        preds_score.append(pred_score)

            #output_dir
            prediction_chunk_path=os.path.join(prediction_test_dir,"test_eRisk_depression_"+str(chunk_num)+".txt")
            pred_user_score = list(zip(users_id, preds_score))
            pred_user_score_df = pd.DataFrame(pred_user_score)
            pred_user_score_df.to_csv(prediction_chunk_path, sep='\t', index=False, header=False)
            print ("done chunk ." + str(chunk_num))
