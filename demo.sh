#DCRSR-train

#python3 main.py --n_GPUs 1  --batch_size 1  --model DCRSR --scale 2 --save DCRSR_x2_3_3_64_shuffle  --n_dresblocks 3  --patch_size 64  --n_resblocks 3 --n_feats 64 --res_scale 0.1 --reset  
 
#DCRSR-test
python3 main.py --test_only  --batch_size 1  --model DCRSR --scale 2  --pre_train experiment/DCRSR_x2_3_3_128/model/model_best.pt --n_dresblocks 3  --patch_size 64  --n_resblocks 3 --n_feats 128 --res_scale 0.1 

#MDFSRCNN-train
#python3 main.py --n_GPUs 2  --batch_size 1  --model MDFSRCNN --scale 2 --save 3DFSRCNN_x2   --reset --loss 1*MSE

#MDFSRCNN-test
#python3 main.py --test_only  --batch_size 1  --model MDFSRCNN --scale 2  --pre_train experiment/3DFSRCNN_x2/model/model_best.pt  --loss 1*MSE

#NDFSRCNN-train
#python3 main.py --n_GPUs 2  --batch_size 1  --model NDFSRCNN --scale 2 --save 2DFSRCNN_x2   --reset --loss 1*MSE

#NDFSRCNN-test
#python3 main.py --test_only  --batch_size 1  --model NDFSRCNN --scale 2  --pre_train experiment/2DFSRCNN_x2/model/model_best.pt  --loss 1*MSE


#EDSR-train
#python3 main.py --model EDSR --batch_size 1 --scale 2 --save edsr_x2_32_256   --patch_size 64  --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset 

#EDSR-test
#python3 main.py --test_only  --batch_size 1  --model EDSR --scale 2  --pre_train experiment/edsr_x2_16_64/model/model_best.pt  --patch_size 64  --n_resblocks 16 --n_feats 64 --res_scale 0.1 

#DCSRN-train
#python3 main.py --n_GPUs 1  --batch_size 2  --model DCSRN --scale 2 --save DCSRN  --lr 1e-4 --loss 1*MSE  --patch_size 64  --reset --input_large 

#DCSRN-test
#python3 main.py --test_only  --batch_size 1  --model DCSRN --scale 2  --pre_train experiment/DCSRN/model/model_best.pt  --lr 1e-4 --loss 1*MSE  --patch_size 64 --input_large 
