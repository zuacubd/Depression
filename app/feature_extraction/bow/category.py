import re
import ssl
import unicodedata


def get_category_tokens():
    category_tokens = []
    filename = "criterias/categories/categories.txt"

    f = open(filename, "r", encoding="utf8")
    for line in f:
        category_tokens.append(line[:-1])
    f.close()

    return category_tokens


def get_sentence_to_category():
    sentence_to_category = {}
    filename = "criterias/categories/2018_sentence_to_categories.txt"

    f = open(filename, "r", encoding="utf8")

    for line in f:
        parts = line.rstrip().split('\t')
        sentence = parts[2]
        #print (len(parts))
        #print (sentence)

        if len(parts)==3:
            category = ''
        else:
            category = parts[3]

        #print (category)
        sentence_to_category[sentence] = category

    f.close()

    return sentence_to_category


class Category:

    def __init__(self):
        self.category_items = get_category_tokens()
        self.sentence_to_category = get_sentence_to_category()
        self.text = ""
        self.category_vocab = {}

    def build_category_vocab(self):
        for item in self.category_items:
            self.category_vocab[item] = 0

    def generate_category_features(self):

        category_labels = self.sentence_to_category[self.text]
        category_labels_list = []
        if len(category_labels) > 0:
            category_labels_list = category_labels.split(";")

        #print (self.text)
        #print (category_labels)
        #print (len(category_labels_list))

        clean_category_labels = []
        for i in range(len(category_labels_list)):
            label = category_labels_list[i]

            clean_label = ''
            parts = label.split('/')
            if len(parts) == 2:
                clean_label = parts[1]
            elif len(parts) >= 2:
                clean_label = parts[1] + "/" + parts[2]

            clean_category_labels.append(clean_label)

        for i in range(len(clean_category_labels)):
            item = clean_category_labels[i]
            self.category_vocab[item] += 1


    def get_categories(self, text):
        self.text = text
        self.build_category_vocab()
        category_features = self.generate_category_features()
        return self.category_vocab
