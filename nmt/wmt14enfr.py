"""
Data iterator for text datasets that are used for translation model.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy 

import os, gc

import tables
import copy
import logging

import threading
import Queue

import collections
from tm_dataset import PytablesBitextFetcher, PytablesBitextIterator


def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen != None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x,seqs_y)):
        s_x[numpy.where(s_x >= n_words_src-1)] = 1
        s_y[numpy.where(s_y >= n_words-1)] = 1
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.
        y[:lengths_y[idx],idx] = s_y
        y_mask[:lengths_y[idx]+1,idx] = 1.

    return x, x_mask, y, y_mask

def load_data(batch_size=128):
    ''' 
    Loads the dataset
    '''

    path_src = '/data/lisatmp3/chokyun/wmt14/parallel-corpus/en-fr/parallel.en.shuf.h5'
    path_trg = '/data/lisatmp3/chokyun/wmt14/parallel-corpus/en-fr/parallel.fr.shuf.h5'

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'

    train = PytablesBitextIterator(batch_size, path_trg, path_src, use_infinite_loop=False)
    valid = None
    test = None

    return train, valid, test


