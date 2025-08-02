from __future__ import print_function

import sys
sys.path.append('/mnt/nfs/data/home/2220221950/zxj/Circuitnet3')
# from models.DeepGCN import pvig_ti_224_gelu

import os
import os.path as osp
import json
import numpy as np
from scipy.constants import golden

from tqdm import tqdm

from datasets.build_dataset import build_dataset
# from routability_ir_drop_prediction.utils.metrics import calculate_average_mae
from utils.metrics import build_metric, build_roc_prc_metric, calculate_average_mae
from models.build_model import build_model
from utils.configs import Parser


def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    # model = pvig_ti_224_gelu()
    if not arg_dict['cpu']:
        model = model.cuda()

    # Build metrics
    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0 for k in arg_dict['eval_metric']}

    count = 0
    goldens = []
    predicts = []

    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if arg_dict['cpu']:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)
            # 添加
            goldens.append(target.cpu().numpy())
            predicts.append(prediction.squeeze(1).cpu().detach().numpy())

            for metric, metric_func in metrics.items():
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

            if arg_dict['plot_roc']:
                save_path = osp.join(arg_dict['save_path'], 'test_result')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = osp.splitext(osp.basename(label_path[0]))[0]
                save_path = osp.join(save_path, f'{file_name}.npy')
                output_final = prediction.float().detach().cpu().numpy()
                #print(output_final)
                np.save(save_path, output_final)
                count += 1

            bar.update(1)

    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    # 添加
    average_mae = calculate_average_mae(goldens, predicts)
    print(f"=====> Average MAE: {average_mae:.4f}")

    # eval roc & prc 以及最终的平均 F1 值
    if arg_dict['plot_roc']:
        roc_metric, prc_metric, avg_f1 = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC: {:.4f}".format(roc_metric))
        print("\n===> Average F1 Score: {:.4f}".format(avg_f1))


if __name__ == "__main__":
    test()
