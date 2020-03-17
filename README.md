# nurion4hep/HEP-CNN
Reproduce HEP-CNN by NERSC at KISTI Nurion

```
git clone https://github.com/nurion4hep/HEP-CNN
```

## Data preparation
A. Download dataset from NERSC (recommended for computing performance checks)
B. Run Delphes simulator + projection script provided by NERSC

### Option A: Download ready-to-analyze datasets
```
./scripts/download_NERSC.sh
```

### Option B (available only for CMS internal users): Generate samples from CMS MC
Requirements: A Linux workstation configured with CVMFS, CMSSW

*Step1: Install packages* \\
```
./scripts/install_CMSDelphes.sh
```

*Step2: Run the Delphes*\\
Example: extract prunedGenParticles+packedGenParticles from CMS MiniAOD and run the Delphes simulator.
```
cd Delphes
./DelphesCMSFWLite cards/delphes_card_CMS.tcl ../DELPHES.root root://cms-xrd-global.cern.ch//store/mc/RunIISummer16MiniAODv2/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/FED04F50-A3BE-E611-A893-0025900E3502.root
cd ..
```
You will have DELPHES.root 

*Step3: project on a MxM "image"* \\
We use NERSC's script to convert Delphes root files to hdf5 with image projections.
- File names should be in a form of SAMPLENAME-SUFFIX.root
- SAMPLENAME should be in the cross section table, config/DelphesXSec
  - RPV10\_1400\_850
  - QCDBkg\_JZ3\_160\_400
  - QCDBkg\_JZ4\_400\_800
  - QCDBkg\_JZ5\_800\_1300
  - QCDBkg\_JZ6\_1300\_1800
  - QCDBkg\_JZ7\_1800\_2500
  - QCDBkg\_JZ8\_2500\_3200
  - QCDBkg\_JZ9\_3200\_3900
  - QCDBkg\_JZ10\_3900\_4600
  - QCDBkg\_JZ11\_4600\_5300
  - QCDBkg\_JZ12\_5300\_7000
- List of files in a txt file

Example:
```
mv DELPHES.root RPV10_1400_850-xxxx.root
echo ../RPV10_1400_850-xxxx.root > fileList.txt ## NOTE the relative path
```

```
git clone https://github.com/eracah/atlas_dl
cd atlas_dl
./scripts/prepare_data.py --input-type delphes --output-h5 ../../data.h5 --bins 64 ../fileList.txt 
```

## Prepresess HEP-CNN images
For a efficient data analysis, a pre-processing step is applied before the full training.
In this step, we join detector images scattered to different hdf5 groups into one, multi-channel images and store with optimal compression options. 

Note1: default image format is CPU/Tensorflow-friendly NHWC. 
For the pytorch, NCHW is the default image format - currently, our pytorch dataset implementataion transposes the matrix NHWC to NHWC - to be modified.

Note2: Do we really need suffix "val" in the h5 data? (I'm just following original Keras codes)

Note3: Curcular padding, log scaling is turned off for the computing performance testing.

```
## Assume we are back under HEP-CNN directory
mkdir -p data/NERSC_preproc/
python scripts/preprocess.py -i data/NERSC/train.h5 -o data/NERSC_preproc/train.h5 --format NHWC
python scripts/preprocess.py -i data/NERSC/val.h5 -o data/NERSC_preproc/val.h5 --format NHWC --suffix val
python scripts/preprocess.py -i data/NERSC/test.h5 -o data/NERSC_preproc/test.h5 --format NHWC --suffix val
```

## Run the training
Almost everything's ready under 'run' directory.

NOTE: pytorch training code is to be updated (multi-node emulation part)

```
cd run
SELECT=128 BATCH=8 NTHREADS=64 MPIPROC=128 ./run_torch_nurion.sh
SELECT=256 BATCH=8 NTHREADS=64 MPIPROC=256 ./run_torch_nurion.sh
SELECT=512 BATCH=8 NTHREADS=64 MPIPROC=512 ./run_torch_nurion.sh
SELECT=1024 BATCH=8 NTHREADS=64 MPIPROC=1024 ./run_torch_nurion.sh
SELECT=2048 BATCH=8 NTHREADS=64 MPIPROC=2048 ./run_torch_nurion.sh
SELECT=4096 BATCH=8 NTHREADS=64 MPIPROC=4096 ./run_torch_nurion.sh
SELECT=8192 BATCH=8 NTHREADS=64 MPIPROC=8192 ./run_torch_nurion.sh
```

## Monitor the CPU performance
```
python drawPerfHistory.py perf_nurion_KNL_torch/SELECT_128__MPIPROC_128__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_256__MPIPROC_256__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_512__MPIPROC_512__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_1024__MPIPROC_1024__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_2048__MPIPROC_2048__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_4096__MPIPROC_4096__THREADS_64__BATCH_8 \
                          perf_nurion_KNL_torch/SELECT_8192__MPIPROC_8192__THREADS_64__BATCH_8
```
