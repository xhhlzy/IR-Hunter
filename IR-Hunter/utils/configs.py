import argparse
import os
import sys

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')

        self.parser.add_argument('--save_path', default='work_dir/congestion_gpdl/')
    
        self.parser.add_argument('--pretrained', default=None)

        self.parser.add_argument('--max_iters', default=200000)
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', action='store_true')
        self.get_remainder()
        
    def get_remainder(self):
        if self.parser.parse_args().task == 'irdrop_mavi':
            self.parser.add_argument('--dataroot', default='/mnt/nfs/data/home/2220221950/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=4)

            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.9885)

        elif self.parser.parse_args().task == 'FCN':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.9885)


        elif self.parser.parse_args().task == 'VCAttUNet':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='VCAttUNet')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'mae', 'cc'])
            self.parser.add_argument('--threshold', default=0.9885)

        elif self.parser.parse_args().task == 'VCAttUNet_Large':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='VCAttUNet_Large')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'mae', 'cc'])
            self.parser.add_argument('--threshold', default=0.9885)

        elif self.parser.parse_args().task == 'IREDGe':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='IREDGe')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'mae', 'cc'])
            self.parser.add_argument('--threshold', default=0.9885)

        elif self.parser.parse_args().task == 'powernet_change':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='powernet_change')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'mae', 'cc'])
            self.parser.add_argument('--threshold', default=0.9885)


        elif self.parser.parse_args().task == 'Hunter':
            self.parser.add_argument('--dataroot', default='/media/user/E5A765C423DB74CB/zxj/IR_drop_1.0')
            self.parser.add_argument('--ann_file_train', default='./files/train_N28.csv')
            self.parser.add_argument('--ann_file_test', default='./files/test_N28.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=8)

            self.parser.add_argument('--model_type', default='Hunter')
            self.parser.add_argument('--in_channels', default=1)# 24
            self.parser.add_argument('--out_channels', default=4)# 2
            self.parser.add_argument('--lr', default=2e-4)# 2e-4
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM', 'mae', 'cc'])
            self.parser.add_argument('--threshold', default=0.9885)


        else:
            raise ValueError
