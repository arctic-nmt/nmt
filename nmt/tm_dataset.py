"""
Data iterator for text datasets that are used for translation model.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy as np

import os, gc
import weakref

import tables
import copy
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)

class PytablesBitextFetcher(threading.Thread):
    def __init__(self, parent, start_offset):
        threading.Thread.__init__(self)
        self.parent = parent
        self.start_offset = start_offset

    def run(self):
        diter = self.parent

        driver = None
        if diter.can_fit:
            driver = "H5FD_CORE"

        if tables.__version__[0] == '2':
            target_table = tables.openFile(diter.target_file, 'r')
            target_data, target_index = (target_table.getNode(diter.table_name),
                                         target_table.getNode(diter.index_name))
        else:
            target_table = tables.open_file(diter.target_file, 'r', driver=driver)
            target_data, target_index = (target_table.get_node(diter.table_name),
                                         target_table.get_node(diter.index_name))

        if tables.__version__[0] == '2':
            source_table = tables.openFile(diter.source_file, 'r')
            source_data, source_index = (source_table.getNode(diter.table_name),
                                         source_table.getNode(diter.index_name))
        else:
            source_table = tables.open_file(diter.source_file, 'r', driver=driver)
            source_data, source_index = (source_table.get_node(diter.table_name),
                                         source_table.get_node(diter.index_name))

        assert source_index.shape[0] == target_index.shape[0]
        data_len = source_index.shape[0]

        offset = self.start_offset
        if offset == -1:
            offset = 0
            if diter.shuffle:
                offset = np.random.randint(data_len)
        logger.debug("{} entries".format(data_len))
        logger.debug("Starting from the entry {}".format(offset))

        while not diter.exit_flag:
            last_batch = False
            source_sents = []
            target_sents = []
            while len(source_sents) < diter.batch_size:
                if offset == data_len:
                    if diter.use_infinite_loop:
                        offset = 0
                    else:
                        last_batch = True
                        break

                slen, spos = source_index[offset]['length'], source_index[offset]['pos']
                tlen, tpos = target_index[offset]['length'], target_index[offset]['pos']
                offset += 1

                if slen > diter.max_len or tlen > diter.max_len:
                    continue
                source_sents.append(source_data[spos:spos + slen].astype(diter.dtype))
                target_sents.append(target_data[tpos:tpos + tlen].astype(diter.dtype))

            if len(source_sents):
                diter.queue.put([int(offset), source_sents, target_sents])
            if last_batch:
                diter.queue.put([None])
                source_table.close()
                target_table.close()
                return

class PytablesBitextIterator(object):

    def __init__(self,
                 batch_size,
                 target_file=None,
                 source_file=None,
                 dtype="int64",
                 table_name='/phrases',
                 index_name='/indices',
                 can_fit=False,
                 queue_size=1000,
                 cache_size=1000,
                 shuffle=True,
                 use_infinite_loop=True,
                 max_len=1000):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.exit_flag = False

    def start(self, start_offset=0):
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = PytablesBitextFetcher(self, start_offset)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        batch = self.queue.get()
        if not batch:
            raise StopIteration
        if len(batch) < 2:
            raise StopIteration
        self.next_offset = batch[0]
        return batch[1], batch[2]

