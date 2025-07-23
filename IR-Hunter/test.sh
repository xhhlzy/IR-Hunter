#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
alias ll='ls -al'  # 快捷键

# redirct stdout&stderr into build.log


# load module
module load anaconda/anaconda3-2022.10
module load cuda/11.8.0
module load gcc
echo "CUDA Version from nvcc"
nvcc --version
# create new conda environment
# conda create -n szx_net_delay

source activate zxj
conda env list >> "/mnt/nfs/data/home/2220221950/zxj/Circuitnet3_1/test.log" 2>&1
nvidia-smi >> "/mnt/nfs/data/home/2220221950/zxj/Circuitnet3_1/test.log" 2>&1

# run and test
cd /mnt/nfs/data/home/2220221950/zxj/Circuitnet3_1/ >> "/mnt/nfs/data/home/2220221950/zxj/Circuitnet3_1/test.log" 2>&1
python train.py --task irdrop_mavi --save_path work_dir/irdrop_mavi/ > train_oneGPUtest.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_120000.pth --save_path work_test/irdrop_mavi1/ --plot_roc > test_oneGPUtest0.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_130000.pth --save_path work_test/irdrop_mavi2/ --plot_roc > test_oneGPUtest1.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_140000.pth --save_path work_test/irdrop_mavi3/ --plot_roc > test_oneGPUtest2.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_150000.pth --save_path work_test/irdrop_mavi4/ --plot_roc > test_oneGPUtest3.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_160000.pth --save_path work_test/irdrop_mavi5/ --plot_roc > test_oneGPUtest4.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_170000.pth --save_path work_test/irdrop_mavi6/ --plot_roc > test_oneGPUtest5.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_180000.pth --save_path work_test/irdrop_mavi7/ --plot_roc > test_oneGPUtest6.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_190000.pth --save_path work_test/irdrop_mavi8/ --plot_roc > test_oneGPUtest7.log 2>&1
python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_200000.pth --save_path work_test/irdrop_mavi9/ --plot_roc > test_oneGPUtest8.log 2>&1