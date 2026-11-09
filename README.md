# 3D-medical-image-segmentation
Methods for Optimizing the Performance of 3D Medical Image Segmentation



## Setup, preprocessing, training and inference commands

Tested with:
* Python 3.13
* CUDA 13.2
* torch 2.13.0

### Step 1: Activate a virtual env
* python >= 3.13, torch >= 2.13.0, CUDA >= 13.2.2

### Step 2: Install the custom nnUNet version
* This version provides Muon trainers, lowres postprocessing, and Hann weighing
```
cd nnUNet
pip install .
cd ..
```

### Step 3: Download datasets and prepare them in the nnUNet format
* Any dataset can be used as long as it is prepared in the nnUNet format
* Check the official nnUNet documentation, or [dataset_format.md](https://github.com/ancestor-mithril/nnUNet/blob/cff21c3ac847403580671084d65c7734ff97ff24/documentation/dataset_format.md)
* TotalSegmentator small can be downloaded from https://zenodo.org/records/10047263.
* TopBrain 2025 can be downloaded from https://zenodo.org/records/21972006.
* Each dataset must have a unique id, DATASET_ID, according to the nnUNet documentation.

### Step 4: Planning and preprocessing
* The command must be used for all DATASET_IDs
* Training can be done only after a dataset was preprocessed
```
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetResEncUNetLPlans_torchres_8 -c 3d_fullres
```

### Step 5: Training
* Training should be done with the TRAINER, which can be 
    * Default trainer: `nnUNetTrainer` (1000 epochs), `nnUNetTrainer_250epochs`, `nnUNetTrainer_500epochs`, `nnUNetTrainer_750epochs`
    * Muon trainer: `nnUNetTrainerMuon` (1000 epochs), `nnUNetTrainerMuon_250`, `nnUNetTrainerMuon_500`, `nnUNetTrainerMuon_750`
* A model must be trained for each FOLD, and FOLD can be 0, 1, 2, 3 or 4
```
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER --c DATASET_ID 3d_fullres FOLD
```

### Step 6: Inference
* Inference can be done only after a TRAINER has been trained on a DATASET_ID for a given FOLD
* Inference needs an INPUT folder with the input images, and an OUTPUT folder where the predictions will be written
* Input images must follow the nnUNet format
* STEP_SIZE is the sliding-window step size to be used for the inference; can be 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 or 1.0; use 0.5 for the default behavior.

```
export PYTHONOPTIMIZE=1
export PERF_LOGGER=1
export nnUNet_compile=1
export copy_elision=1

# Default inference, Gaussian
nn_resampling_nn=0 USE_GAUSSIAN=1 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# Default inference, Hann
nn_resampling_nn=0 USE_GAUSSIAN=0 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# Lowres inference, Gaussian
nn_resampling_nn=1 USE_GAUSSIAN=1 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# Lowres inference, Hann
nn_resampling_nn=1 USE_GAUSSIAN=0 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE
```

### Step 7: Calculating metrics
* Calculate the Macro-averaged Global Dice score between the OUTPUT folders and folders with GT. 
* You can use [label-metrics](https://github.com/fii-optim-lab/label-metrics) or [dice-score-3d](https://github.com/ancestor-mithril/dice-score-3d).
* For measuring the inference time, use tools such as `/usr/bin/time`.