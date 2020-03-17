#!/bin/bash

#PBS -V
#PBS -N tf114_hcepnn
#PBS -q normal
#PBS -W sandbox=PRIVATE
#PBS -A etc
#PBS -l select=1:ncpus=68:mpiprocs=1:ompthreads=64
#PBS -l walltime=01:00:00

#module load gcc/7.2.0 openmpi/3.1.0 craype-mic-knl tensorflow/1.12.0 hdf5-parallel/1.10.2
source /apps/compiler/intel/19.0.4/impi/2019.4.243/intel64/bin/mpivars.sh relase_mt
source /apps/applications/miniconda3/etc/profile.d/conda.sh
#conda activate tf_v1.13
conda activate /scratch/hpcaia02/conda/tf_v1.14

export HDF5_USE_FILE_LOCKING='FALSE'
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_AFFINITY=granularity=fine,compact
export KMP_SETTINGS=1
export CUDA_VISIBLE_DEVICES=""

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="hfi1_0:1"
export LD_LIBRARY_PATH=/opt/pbs/lib:$LD_LIBRARY_PATH
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=64

[ _$MPIPROC == _ ] && MPIPROC=1
[ _$NTHREAD == _ ] && NTHREAD=8
[ _$BATCH == _ ] && BATCH=32
[ _$SELECT == _ ] && SELECT=1
OUTDIR=perf_nurion_KNL_keras/SELECT_${SELECT}__MPIPROC_${MPIPROC}__THREADS_${NTHREAD}__BATCH_${BATCH}

[ _$PBS_O_WORKDIR != _ ] && cd $PBS_O_WORKDIR
[ -d $OUTDIR ] || mkdir -p $OUTDIR
mpirun -np $MPIPROC -env OMP_NUM_THREADS $NTHREAD \
    python train_keras.py -o $OUTDIR \
           --epoch 5 --batch $BATCH \
           -t ../data/NERSC_preproc_base/train.h5 -v ../data/NERSC_preproc_base/val.h5 \


