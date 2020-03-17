#!/usr/bin/env python
import numpy as np
import h5py
from tensorflow.keras.preprocessing.image import Iterator

class HEPCNNDataGenerator(Iterator):
    def __init__(self, fileName, batch_size, shuffle=False, seed=1234567, nEvent=-1, syslogger=None):
        if syslogger: syslogger.update(annotation='open file '+fileName)
        self.fileName = fileName
        if fileName.endswith('.h5'):
            data = h5py.File(fileName, 'r', libver='latest', swmr=True)
        elif fileName.endswith('.npz'):
            data = {'all_events':np.load(fileName)}
        suffix = '_val' if 'images_val' in data['all_events'] else ''

        if syslogger: syslogger.update(annotation='read file')
        self.images = data['all_events']['images'+suffix][()]
        self.labels = data['all_events']['labels'+suffix][()]
        self.weights = data['all_events']['weights'+suffix][()]

        if nEvent > 0:
            self.images = self.images[:nEvent]
            self.labels = self.labels[:nEvent]
            self.weights = self.weights[:nEvent]
            if syslogger: syslogger.update(annotation='select events')

        self.shape = self.images.shape
        self.num_batch = int(self.shape[0]/batch_size)
        self.idxs = [(x*batch_size, (x+1)*batch_size) for x in range(self.num_batch)]
        self.idxs[-1] = (self.idxs[-1][0], min(self.idxs[-1][1], self.shape[0]))

        super(HEPCNNDataGenerator, self).__init__(self.shape[0], batch_size, shuffle, seed)

    def __len__(self):
        return self.num_batch

    def __getitem__(self, i):
        idx1, idx2 = self.idxs[i]
        return (self.images[idx1:idx2], self.labels[idx1:idx2], self.weights[idx1:idx2])

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.idxs)
