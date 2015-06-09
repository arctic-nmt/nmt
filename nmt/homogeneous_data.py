import numpy

import itertools
import operator

from tm_dataset import PytablesBitextIterator 

class HomogenousData(PytablesBitextIterator):

    def __init__(self, *args, **kwargs):
        PytablesBitextIterator.__init__(self, *args, **kwargs)
        self.batch_iter = None

    def get_homogenous_batch_iter(self):
        end_of_iter = False
        while True:
            k_batches = 10
            batch_size = self.batch_size
            x = []
            y = []
            for k in xrange(k_batches):
                try:
                    dx, dy = PytablesBitextIterator.next(self)
                except StopIteration:
                    end_of_iter = True
                    break
                if dx == None or dy == None:
                    break
                x += dx
                y += dy
            if len(x) <= 0 or len(y) <= 0:
                raise StopIteration
            lens = numpy.asarray([map(len, x), map(len, y)])
            order = numpy.argsort(lens.max(axis=0)) if k_batches > 1 else numpy.arange(len(x))
            for k in range(k_batches):
                if k * batch_size > len(order):
                    break
                indices = order[k * batch_size:(k + 1) * batch_size]
                yield [[x[ii] for ii in indices], [y[ii] for ii in indices]]

            if end_of_iter:
                raise StopIteration

    def next(self, peek=False):
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter()
        if not self.batch_iter:
            raise StopIteration
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            self.batch_iter = None
            raise StopIteration

        return batch[0], batch[1]

