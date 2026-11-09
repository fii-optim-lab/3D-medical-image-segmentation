# 3D-medical-image-segmentation
Methods for Optimizing the Performance of 3D Medical Image Segmentation

## Requirements

Tested with:
* Python 3.13
* CUDA 13.2
* torch 2.13.0


## Setup, preprocessing, training and inference commands

```
# 1. Activate a virtual env

# 2. Install the custom nnUNet with Muon trainer and postprocessing options
cd nnUNet
pip install .
cd ..

# 3. Download datasets from their source and prepare them in the nnUNet format (check the official nnUNet documentation)

# 4. Planning and preprocessing => use the DATASET_ID of your dataset
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetResEncUNetLPlans_torchres_8 -c 3d_fullres

# 5. Training with the standard trainer on your DATASET_ID
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainer --c DATASET_ID 3d_fullres 0
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainer --c DATASET_ID 3d_fullres 1
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainer --c DATASET_ID 3d_fullres 2
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainer --c DATASET_ID 3d_fullres 3
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainer --c DATASET_ID 3d_fullres 4

# 6. Training with the Muon trainer on your DATASET_ID
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainerMuon --c DATASET_ID 3d_fullres 0
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainerMuon --c DATASET_ID 3d_fullres 1
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainerMuon --c DATASET_ID 3d_fullres 2
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainerMuon --c DATASET_ID 3d_fullres 3
nnUNetv2_train -p nnUNetResEncUNetLPlans_torchres_8 -tr nnUNetTrainerMuon --c DATASET_ID 3d_fullres 4

# 7. Training for fewer epochs: use nnUNetTrainer_250epochs, nnUNetTrainer_500epochs, nnUNetTrainer_750epochs instead of nnUNetTrainer
# And nnUNetTrainerMuon_250, nnUNetTrainerMuon_500, nnUNetTrainerMuon_750 instead of nnUNetTrainerMuon

# 8. Inference environment variables
export PYTHONOPTIMIZE=1
export PERF_LOGGER=1
export nnUNet_compile=1
export copy_elision=1

# * INPUT is the folder with the input images
# * OUTPUT is the folder in which segmentations are written
# * DATASET_ID is the dataset id on which the model was trained
# * FOLD is a trained fold, either 0, or 1, or 2, or 3, or 4
# * TRAINER is a trained trainer (e.g. nnUNetTrainerMuon)
# * STEP_SIZE is the sliding-window step size to be used for the inference; use 0.5 for the default behavior

# 9. Default inference, Gaussian
nn_resampling_nn=0 USE_GAUSSIAN=1 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# 10. Default inference, Hann
nn_resampling_nn=0 USE_GAUSSIAN=0 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# 11. Lowres inference, Gaussian
nn_resampling_nn=1 USE_GAUSSIAN=1 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE

# 12. Lowres inference, Hann
nn_resampling_nn=1 USE_GAUSSIAN=0 nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f FOLD -p nnUNetResEncUNetLPlans_torchres_8 -tr TRAINER -npp 0 -nps 0 --not_on_device --continue_prediction -step_size STEP_SIZE
```