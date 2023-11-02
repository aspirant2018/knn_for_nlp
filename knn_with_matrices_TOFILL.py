#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
from math import *

from collections import Counter, defaultdict


class Example:
    """
    An example : 
    vector = vector representation of an object (dict for features with non-null values only)
    gold_class = gold class for this object
    """
    def __init__(self, example_number, gold_class):
        self.gold_class = gold_class
        self.example_number = example_number
        self.vector = {}

    def add_feat(self, featname, val):
        self.vector[featname] = val


class Indices:
    """ class to handle the correspondances from classes to indices and from words to indices correspondances """
    
    def __init__(self):
        self.i2c = []
        self.c2i = {}
        self.i2w = []
        self.w2i = {}

    def add_w(self, wordform):
        if wordform not in self.w2i:
            self.w2i[wordform] = len(self.i2w)
            self.i2w.append(wordform)

    def add_c(self, classlabel):
        if classlabel not in self.c2i:
            self.c2i[classlabel] = len(self.i2c)
            self.i2c.append(classlabel)
            
    def get_word_vocab_size(self):
        return len(self.i2w)

    def get_nb_classes(self):
        return len(self.i2c)
    
    def i_from_c(self, c):
        """ adds the class c if unknown yet, and retrieve index of class c """
        if c in self.c2i:
            return self.c2i[c]
        self.c2i[c] = len(self.i2c)
        self.i2c.append(c)
        return self.c2i[c]

    def i_from_w(self, w, create_new=False):
        """ if word w is already known : returns its index
            otherwise, either register it and return its new index, or returns None, depending on boolean create_new """
        if w in self.w2i:
            return self.w2i[w]
        if not create_new:
            return None
        self.w2i[w] = len(self.i2w)
        self.i2w.append(w)
        return self.w2i[w]


def normalize_row_vectors(X):
    """ returns the normalized version of the row vectors in X
        (each row vector of X divided by its norm)
    """
        
    # TODO

    # getting  the number of rows 
    num_rows = X.shape[0]
    # caluclating the norm 
    norm = np.sqrt(np.sum(X*X,axis=1)).reshape(num_rows,1)
    
    X_normalized = X/norm

    

    return X_normalized 

    

class KNN:
    """
    K-NN for document classification (multiclass)

    members = 

    X_train = matrix of training example vectors
    Y_train = list of corresponding gold classes

    K = maximum number of neighbors to consider

    """
    def __init__(self, X, Y, K=1, weight_neighbors=False, trace=False):
        self.X_train = X   # (nbexamples, d)
        self.Y_train = Y   # list of corresponding gold classes

        # nb neighbors to consider
        self.K = K

        # example vectors divided by their norm
        print("Normalizing row vectors of training set...")
        self.X_train_normalized = normalize_row_vectors(X)

        # if True, the nb of neighbors will be weighted by their similarity to the example to classify
        self.weight_neighbors = weight_neighbors

        self.trace = trace

    def evaluate_on_test_set(self, X_test, Y_test, indices):
        """ Runs the K-NN classifier on examples stored in X_test/Y_test
        for k values ranging from 1 to self.K

        Returns a list of accuracies, for k in the range 1 to K
        (first element = accuracy when k=1, last element = accuracy when k=self.K)
        """
        print("Normalizing row vectors of test set...")
        X_test_normalized = normalize_row_vectors(X_test)

        # TODO
        # at this point X_test_normalized and self.X_train_normalized
        # contain the normalized vectors of test and train examples

        # cos_matrix = ...


        cos_matrix = np.transpose(np.dot(self.X_train_normalized, np.transpose(X_test_normalized)))

        unsorted_list = []
        sorted_list = []
        
        for row in cos_matrix:
            l = row.tolist()
            for i,value in enumerate(l):
                unsorted_list.append((i,value))

            #sorting the each row 
            sorted_list  = sorted(unsorted_list, key=lambda x: x[1], reverse=True)[:3]

            classes = [ self.Y_train[x[0]] for x in sorted_list]
            print(classes)
            
            #print(majority_vote)
            #predicted = majority_vote.most_common(1)
            #print(predicted)
            for i in range(classes.shape[0]):
                classe1 = classes[i].toList()
                majority_vote = Counter(classe1)
                predicted = majority_vote.most_common(1)
                print(predicted)




        


        if self.trace:
            print("Matrix of cosine similarities (rows = test, columns = train):")
            print(cos_matrix)
            

        accuracies = [0 for k in range(self.K)]
        return accuracies


def read_examples(infile, indices=None):
    """ Reads a .examples file and returns a list of Example instances 
    if indices is not None but an instance of Indices, 
    it is updated with potentially new words/indices while reading the examples
    """
    update_indices = (indices != None)

    stream = open(infile)
    examples = []
    example = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if example != None:
                examples.append(example)
            cols = line.split('\t')
            gold_class = cols[3]
            example_number = cols[1]
            example = Example(example_number, gold_class)
            if update_indices:
                # create id for this class if it was not seen before
                indices.add_c(gold_class)
        elif line and example != None:
            (wordform, val) = line.split('\t')
            example.add_feat(wordform, float(val))
            if update_indices:
                # create id for this word if it was not seen before
                indices.add_w(wordform)

    
    if example != None:
        examples.append(example)
    return examples


def build_matrices(examples, indices):
    """ turn the examples into a X matrix (nb-examples, length of vectors)
    and a Y list of gold classes """

    # TODO
    # n the number of rows , m the number of columns
    n = len(examples)
    m = indices.get_word_vocab_size()

    # initializing the matrix 
    X = np.zeros((n,m))
    Y = []

    for i,example in enumerate(examples):

        #each indice i has a class in the Y list
        Y.append(example.gold_class)
        
        for key,value in example.vector.items():
            #print(key)
            #print(value)
            
            if key in indices.w2i: # the word is in the vocabulary
                j = indices.w2i[key]
                X[i,j]=value
            else:
                # the word is not in the vocabulary 
                pass

    return (X, Y)
    


usage = """ CLASSIFIEUR de DOCUMENTS, de type K-NN

  prog [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE et TEST_FILE sont au format *.examples

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='fichier d\'exemples, utilisés comme voisins', default=None)
parser.add_argument('test_file', help='fichier d\'exemples, utilisés pour évaluation du K-NN', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Hyperparamètre K : le nombre max de voisins à considérer pour la classification (toutes les valeurs de 1 a k seront testées). Default=1')
parser.add_argument('-v', '--trace',action="store_true",default=False,help="A utiliser pour déclencher un mode verbeux. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="Pondération des voisins. Default=False")
args = parser.parse_args()




#------------------------------------------------------------
# Loading training examples :
#    we pass an empty Indices instance,
#    which will be filled with word --> id and id--> word correspondences

indices = Indices()
train_examples = read_examples(args.train_file, indices)

# Loading test examples : we don't add any new word to the vocabulary
test_examples = read_examples(args.test_file, indices=None)

#------------------------------------------------------------
# Organize the data into two matrices for document vectors
#                   and two lists for the gold classes
(X_train, Y_train) = build_matrices(train_examples, indices)
(X_test, Y_test) = build_matrices(test_examples, indices)

myclassifier = KNN(X = X_train,
                   Y = Y_train,
                   K = args.k,
                   weight_neighbors = args.weight_neighbors,
                   trace=args.trace)

print("Evaluating on test...")
accuracies = myclassifier.evaluate_on_test_set(X_test, Y_test, indices)


