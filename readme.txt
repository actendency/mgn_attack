#测试命令
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --datadir ../../Dataset/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save adam_1 --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad --test_only
#目标图片路径
Market-1501-v15.09.15\attack
#生成对抗样本路径
MGN-pytorch\adv
#对抗样本测试路径
Market-1501-v15.09.15\attack