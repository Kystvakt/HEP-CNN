#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys, os
import subprocess
import csv
import math

import tensorflow as tf

try:
    import horovod.tensorflow.keras as hvd
except:
    hvd = None

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
config = tf.compat.v1.ConfigProto()
## From Nurion user guide, intra=n_physical_core, inter=1, keep inter*intra <= n_phisical_core
config.intra_op_parallelism_threads = nthreads ## for independent graph computations
config.inter_op_parallelism_threads = 1 ## for operations which can run in parallel such as matmul or reduction
## From TF performance manual page, intra=inter=n_physical_core or n_logical_core
#config.intra_op_parallelism_threads = nthreads
#config.inter_op_parallelism_threads = nthreads
sess = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--ntrain', action='store', type=int, default=-1, help='Number of events for training')
parser.add_argument('--ntest', action='store', type=int, default=-1, help='Number of events for test/validation')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-t', '--trndata', action='store', type=str, required=True, help='input file for training')
parser.add_argument('-v', '--valdata', action='store', type=str, required=True, help='input file for validation')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--lr', action='store', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--noEarlyStopping', action='store_true', help='do not apply Early Stopping')

args = parser.parse_args()

hvd_rank, hvd_size = 0, 1
if hvd:
    hvd.init()
    hvd_rank = hvd.rank()
    hvd_size = hvd.size()
    print("Horovod is available. (rank=%d size=%d)" % (hvd_rank, hvd_size))

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
weightFile = os.path.join(args.outdir, 'weight_%d.h5' % hvd_rank)
predFile = os.path.join(args.outdir, 'predict_%d.npy' % hvd_rank)
trainingFile = os.path.join(args.outdir, 'training_%d.csv' % hvd_rank)
resourceByCPFile = os.path.join(args.outdir, 'resourceByCP_%d.csv' % hvd_rank)
resourceByTimeFile = os.path.join(args.outdir, 'resourceByTime_%d.csv' % hvd_rank)

proc = subprocess.Popen(['python', '../scripts/monitor_proc.py', '-t', '1',
                        '-o', resourceByTimeFile, '%d' % os.getpid()],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.times = []
    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")
from monitor_proc import SysStat
class SysStatHistory(tf.keras.callbacks.Callback, SysStat):
    def __init__(self, pid):
        SysStat.__init__(self, pid, fileName=resourceByCPFile)
    def on_epoch_begin(self, batch, logs):
        self.update(annotation='epoch_begin')
    def on_epoch_end(self, batch, logs):
        self.update(annotation='epoch_end')
    def on_batch_end(self, batch, logs):
        self.update()
sysstat = SysStatHistory(os.getpid())
sysstat.update(annotation="start_logging")

sys.path.append("../python")
from HEPCNN.keras_dataGenerator import HEPCNNDataGenerator as DataLoader
trn_dataLoader = DataLoader(args.trndata, args.batch, shuffle=False, nEvent=args.ntrain, syslogger=sysstat)
if hvd:
    val_dataLoader = DataLoader(args.valdata, args.batch, shuffle=False, nEvent=args.ntest, syslogger=sysstat)
else:
    val_dataLoader = DataLoader(args.valdata, 512, shuffle=False, nEvent=args.ntest, syslogger=sysstat)
steps_per_epoch  = len(trn_dataLoader)//hvd_size
validation_steps = len(val_dataLoader)//hvd_size
if hvd_size > 1: validation_steps *= 3

## Build model
from HEPCNN.keras_model_default import MyModel
model = MyModel(trn_dataLoader.shape[1:])

optm = tf.keras.optimizers.Adam(args.lr*hvd_size)
if hvd: optm = hvd.DistributedOptimizer(optm)

model.compile(
      optimizer=optm,
      loss='binary_crossentropy',
      metrics=['accuracy']
)
model.summary()
sysstat.update(annotation="modelsetup_done")

if not os.path.exists(weightFile):
    try:
        timeHistory = TimeHistory()
        sysstat.update(annotation="train_start")
        callbacks = [
            timeHistory, sysstat,
        ]
        if not args.noEarlyStopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'))
        if hvd: callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        if hvd_rank == 0:
            callbacks.extend([
                tf.keras.callbacks.TensorBoard(log_dir=args.outdir, histogram_freq=1, write_graph=True, write_images=True),
                tf.keras.callbacks.ModelCheckpoint(weightFile, monitor='val_loss', verbose=True, save_best_only=True),
            ])
        #history = model.fit(trn_dataLoader.images, trn_dataLoader.labels, sample_weight=trn_dataLoader.weights,
        #                    validation_data = (val_dataLoader.images, val_dataLoader.labels, val_dataLoader.weights),
        #                    epochs=args.epoch, batch_size=args.batch,
        #                    verbose=1,
        #                    shuffle=False,
        #                    #shuffle='batch',
        #                    #shuffle=True,
        #                    callbacks = callbacks)
        history = model.fit_generator(generator=trn_dataLoader,
                                      validation_data = val_dataLoader,
                                      steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
                                      epochs=args.epoch, verbose=1 if hvd_rank == 0 else 0, workers=4,# use_multiprocessing=True,
                                      callbacks = callbacks)
        sysstat.update(annotation="train_end")

        history.history['time'] = timeHistory.times[:]
        with open(trainingFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.history.keys()
            writer.writerow(keys)
            for row in zip(*[history.history[key] for key in keys]):
                writer.writerow(row)
        sysstat.update(annotation="wrote_logs")

    except KeyboardInterrupt:
        print("Training finished early")

if hvd_rank == 0:
    model.load_weights(weightFile)
    pred = model.predict(val_dataLoader.images, verbose=1, batch_size=args.batch)

    np.save(predFile, pred)
    sysstat.update(annotation="saved_model")
