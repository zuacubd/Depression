import collections
import csv
import os
import sys
import tensorflow as tf
import argparse
import random
import numpy as np
import pandas as pd
import re
import emoji
import xml.etree.ElementTree as et
from datetime import datetime
from dateutil.parser import parse
import pickle
import gensim.models as g
from gensim.utils import simple_preprocess
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)




########################################################
#                      eRisk data loading
########################################################

def get_user_data_from_xml_file(filePath, classval, chunkNumber = -1):

	tree = et.parse(filePath)
	root = tree.getroot()
	userId = root.find('ID').text
	XMLWritings = root.findall('WRITING')

	writings = []

	for w in XMLWritings:
		title = w.find('TITLE').text.replace('\n', '. ').strip(' ')
		date = datetime.strptime(w.find('DATE').text.strip(' '), "%Y-%m-%d %H:%M:%S")
		text = w.find('TEXT').text.replace('\n', '. ').strip(' ')
		
		writings.append([title, date, text])

	u = {
		"chunk" : chunkNumber,
		"class" : classval,
		"uid" : userId,
		"data": [{"chunk" : chunkNumber, "writings" : writings}]
	}

	return u
 
def get_users_data_from_chunks(chunks, classval='?', chunksDirPath=""):
	users_chunks_data = []
	numchunks = 0
	for i in chunks:
		chunkPath = chunksDirPath+"chunk "+str(i)+"/"
		for file in os.listdir(chunkPath):
			if file.endswith('.xml'):
				users_chunks_data.append(get_user_data_from_xml_file(chunkPath+file, classval, i))
		numchunks += 1

		#print("CHUNK %d LOADED :) (%i entities)"%(i,int(len(users_chunks_data)/numchunks)))

	return get_merged_users_data(users_chunks_data)

def get_merged_users_data(users_chunks_data):
	ids = list(map(lambda e: e["uid"], users_chunks_data))

	ids = list(set(ids))

	merged = []

	for uid in ids:
		user_chunks = get_user_chunks(uid, users_chunks_data)
		u = {
			"uid" : uid,
			"class" : user_chunks[0]["class"],
			"data" : []
		}
		for ch in user_chunks:
			u["data"] += ch["data"]
		merged.append(u)

	return merged

def get_user_chunks(uid, users_chunks_data):
	chunks_data = list(filter(lambda e: e["uid"] == uid, users_chunks_data))
	return chunks_data

def load_train_neg_pos(chunks):
	negs = get_users_data_from_chunks(chunks, 'n', NEG_DIR)
	poss = get_users_data_from_chunks(chunks, 'p', POS_DIR)

	return negs, poss

def load_train_data(chunks):
	negs = get_users_data_from_chunks(chunks, 'n', NEG_DIR)
	poss = get_users_data_from_chunks(chunks, 'p', POS_DIR)

	pos_neg_users = negs + poss

	return pos_neg_users


def load_test_data(chunks):
	users = get_users_data_from_chunks(chunks, '?', TEST_DIR)

	return users


# Attribute access
def merge_writings(user):
	writings = []

	for chunk in user["data"]:
		writings += chunk["writings"]

	return writings


#function vectorizin data
def vectorize(data):
    vector_writing = []
    for user in data:
        m = merge_writings(user)
        post_vector_list = []
        for writing in m:
            if writing[0] =="":
                text = writing[2]
            else:
                if writing[2] !="":
                    text = writing[0]+". "+writing[2]
                else:
                    text = writing[0]
            #input_ids = torch.tensor(tokenizer.encode(text,pad_to_max_length=True,truncation_strategy='only_first', max_length=512)).unsqueeze(0)  # Batch size 1
            #with torch.no_grad():
            #    outputs = model(input_ids)
            #post_vector_list.append(outputs[0][0][0].detach().cpu().numpy())
            #break
            encoded_input = tokenizer(text,truncation=True,truncation_strategy='only_first',
                                      max_length=128,return_tensors='pt')
            output = model(**encoded_input)
            #vectors = []
            text_vect = output[0][0][0].detach().cpu().numpy()
            #for ldx in bert_layers:
            #    vectors.append(output[0][0][ldx].detach().cpu().numpy())
            #print (vectors)
            #text_vect = np.mean(vectors, axis=0)
            #print (text_vect)
            post_vector_list.append(text_vect)
        #print(len(np.mean(post_vector_list,axis=0)))
        u={
            "uid" : user["uid"],
            "class" : user["class"],
            "vector" : np.mean(post_vector_list,axis=0),
            "number_writing" : len(m)

        }
        vector_writing.append(u)
    return vector_writing


if __name__ == '__main__':

    ##start the processing here
    print ("Setting models ...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    #model.eval()
    bert_layers = [-1, -2, -3]
    year = int(sys.argv[1])
    POS_DIR = "/projets/sig/mullah/nlp/depression/data/raw/"+str(year)+"/train/positive_examples_anonymous_chunks/"
    NEG_DIR = "/projets/sig/mullah/nlp/depression/data/raw/"+str(year)+"/train/negative_examples_anonymous_chunks/"
    TEST_DIR = "/projets/sig/mullah/nlp/depression/data/raw/"+str(year)+"/test/"

    print ("Loading users data")
    chunks = [i for i in range(1, 11)]
    train_data = load_train_data(chunks)
    print ("Total users in trainset: {}".format(len(train_data)))

    #training dataset
    vector_train = vectorize(train_data)
    print ("Feature extraction done.")

    print ("Writing features ...")
    for vect in vector_train:
        chemin = "/projets/sig/mullah/nlp/depression/features/"+str(year)+"/train/bert/pos/"+vect["uid"]+".csv"

        if vect["class"] == 'n':
            chemin = "/projets/sig/mullah/nlp/depression/features/"+str(year)+"/train/bert/neg/"+vect["uid"]+".csv"

        with open(chemin, 'a+') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(vect["vector"])

    print ("Done")
