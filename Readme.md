Readme.txt
To train and test a network:
demo.sh file: contains commands to train and test DCRSR,DCSRN, FSRCNN. 
Network configuration options:
 --n_resblocks: Number of Residual blocks 
 --n_dresblocks: Number of Dense blocks
 --n_features: Number of feature/ kernels
 --scale: scaling factor
 --model: name of the model architecture and the python file in which the it is coded. Example: --model DCRSR
 --pre_train: set to the path to the saved trained model file.
 --res_scale: set to 0.1. 
 --loss: set to L1. Other options in option.py file.
 --reset: resets the training process
 --resume:resume training from last epoch. set to the epoch number
 --pre_train: when set to a model file. parameters are initiated to values saved in the model file.
 --cpu: use to train on cpu
 --n_GPUs: number of GPUs to train on
 --precision: set to single or half
 --test_only: use it to test the network
 --load: to continue training and restore the previous state of model. Set to the model directory in DCRSR/experiment/
 Other hyperparameters are set in option.py file.

Dataset folders:
LIDC-IDRI: training HR CT slices arranged as per patient-id 
LIDC-IDRI_lanczos2: training LR CT clices  arranged as per patient-id 
test_HR: test HR slices arranged as per patient-id 
test_LR: test LR slices arranged as per patient-id 

Dataset options:
--dir_data: path to data folder for training and testing
--data_train: name of the training set
--data_test: name of the test set (_HR and _LR are appended to the name to identify the HR and LR folders in the code)
--patch_size: set to 64. The 3D volume is cropped into 64x64x64 cubes in MATLAB.
--batch_size: set to 1. Do not change

Results: 
The training results are stored in DCRSR/experiment/ folder for every trained model.
log.txt: file is created both for training with training, validation PSNR, training and validation loss, training time logged in.
PSNR.pdf: Curve of training vs Validation accuracy
loss_L1.pdf: Curve of training vs Validation loss
config.txt: list of set options

The test results are saved in DCRSR/experiment/test folder.
results_test: SR slices arranged as per patient-id
log.txt: test mean PSNR, test mean SSIM, test time logged in. 
config.txt: list of set options
Results options:
--save_models: to save model checkpoint at every epoch. 
--save: name of the model directory where results will be stored.
