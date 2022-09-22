import xml.etree.ElementTree as et
import os
from datetime import datetime
from dateutil.parser import parse
#from setup import *
import re
import pandas as pd

from bs4 import BeautifulSoup

'''
main functions for the new data:
- get_posts_class
- get_new_user_data_from_xml
- get_new_users_data
- get_new_merged_users_data
-
'''

#=======================================================

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
		chunkPath = os.path.join(chunksDirPath, "chunk "+str(i))
		for file in os.listdir(chunkPath):
			if file.endswith('.xml'):
				users_chunks_data.append(get_user_data_from_xml_file(os.path.join(chunkPath, file),
                                                         classval, i))
		numchunks += 1

		#print("CHUNK %d LOADED :) (%i entities)"%(i,int(len(users_chunks_data)/numchunks)))

	return get_merged_users_data(users_chunks_data)


########################################################
#                      eRisk data
########################################################

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

def load_train_data(chunks, NEG_DIR, POS_DIR):
	negs = get_users_data_from_chunks(chunks, 'n', NEG_DIR)
	poss = get_users_data_from_chunks(chunks, 'p', POS_DIR)

	pos_neg_users = negs + poss

	return pos_neg_users


'''
Here we know the real class of test data thanks to test_golden_truth.txt,
without test_golden_truth.txt, we have to use the 'load_test_data' in line 228
'''
def load_test_data(chunks, year, TEST_DIR):
        labels = {}
        test_golden_truth = os.path.join("resources", "eval", str(year), "risk-golden-truth-test.txt")
        #test_golden_truth = "../resources/risk-golden-truth-test_a.txt"
        #test_golden_truth = "../resources/test_golden_truth.txt"

        with open(test_golden_truth) as f:
            for line in f:
                uid, label = line.strip().split(" ")
                if label == "1":
                    label = u'p'
                else:
                    label = u'n'
                if uid not in labels:
                    labels[uid] = label
                else:
                    print("emm, error")
                    exit(-1)

        users = get_users_data_from_chunks(chunks, '?', TEST_DIR)
        re_users = []
        for item in users:
           this_user = {}
           for key,v in item.items():
               if key == u"class":
                   this_user[key] = labels[item[u'uid']]
               else:
                   this_user[key] = v
           re_users.append(this_user)
        return re_users
	#return users

'''
def load_test_data(chunks):
	users = get_users_data_from_chunks(chunks, '?', TEST_DIR)

	return users
'''

# Attribute access
def merge_writings(user):
	writings = []

	for chunk in user["data"]:
		writings += chunk["writings"]

	return writings


#=======================================================
# test function
#if __name__ == '__main__':
#	users_data = get_new_users_data(POSTS_DIR+"posts")

#=======================================================

