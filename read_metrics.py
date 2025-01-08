import os
import json
import torch
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--scene_id', '-S', type=int, default=17)
parser.add_argument('--wname', '-W', type=str, default='normal')
args = parser.parse_args()

# PVG 训练/测试结果文件
# pvg_test_dir = '/data/duantong/mazipei/PVG/eval_output/waymo_nvs/0017085/eval/test_30000_render/metrics.json'
# pvg_train_dir = '/data/duantong/mazipei/PVG/eval_output/waymo_nvs/0017085/eval/train_30000_render/metrics.json'

# 用于存放最终结果的数据结构
data = {
    'dir': [],
    'train_psnr': [],
    'test_psnr': [],
    'num_gaussians': [],
}

# 尝试读取并记录 pvg 的训练/测试结果
try:
    # with open(pvg_test_dir, 'r') as f:
    #     pvg_test_metrics = json.load(f)
    # with open(pvg_train_dir, 'r') as f:
    #     pvg_train_metrics = json.load(f)

    data['dir'].append('pvg')
    data['train_psnr'].append(31.342565)
    data['test_psnr'].append(27.122913)
    # checkpoint_path = '/data/duantong/mazipei/PVG/eval_output/waymo_nvs/0017085/chkpnt30000.pth'
    # (model_params, first_iter) = torch.load(checkpoint_path)
    # num_gaussians = model_params[1].shape[0]
    data['num_gaussians'].append(2848504)

    # opacity = torch.sigmoid(model_params[6])
    # hit = opacity.cpu().detach().numpy()
    # plt.hist(hit, bins=50, color='blue', edgecolor='black', alpha=0.7)
    # plt.yscale('log')
    # plt.title('Histogram of Opacity Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig(os.path.join("histogram", "pvg.png"))  # 保存为当前目录下的 histogram.png
    # plt.close()  # 关闭绘图

except Exception as e:
    print(f"[Warning] Failed to load pvg metrics: {e}")

# Bezier-GS 训练/测试结果所在目录
root_dir = 'eval_output/waymo_nvs'

for sub_dir in sorted(os.listdir(root_dir)):
    if args.wname not in sub_dir:
        continue

    try:
        # 组装文件路径
        train_json_path = os.path.join(root_dir, sub_dir, 'eval', 'metrics_train_30000.json')
        test_json_path = os.path.join(root_dir, sub_dir, 'eval', 'metrics_test_30000.json')
        checkpoint_path = os.path.join(root_dir, sub_dir, 'chkpnt30000.pth')

        # 读取 JSON 指标
        with open(train_json_path, 'r') as f:
            train_metrics = json.load(f)
        with open(test_json_path, 'r') as f:
            test_metrics = json.load(f)

        # 读取 checkpoint，获取模型的 num_gaussians
        (model_params, first_iter) = torch.load(checkpoint_path)
        num_gaussians = model_params[1].shape[0]

        # 将结果依次加入 data
        data['dir'].append(sub_dir)
        data['train_psnr'].append(train_metrics['psnr'])
        data['test_psnr'].append(test_metrics['psnr'])
        data['num_gaussians'].append(num_gaussians)

        opacity = torch.sigmoid(model_params[6])
        hit = opacity.cpu().detach().numpy()
        os.makedirs(os.path.join("histogram", "opacity"), exist_ok=True)
        plt.hist(hit, bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.yscale('log')
        plt.title('Histogram of Opacity Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join("histogram", "opacity", f'./{sub_dir}.png'))  # 保存为当前目录下的 histogram.png
        plt.close()  # 关闭绘图
        os.makedirs(os.path.join("histogram", "distance"), exist_ok=True)
        start_point = model_params[1][:, 0, :]
        end_point = model_params[1][:, -1, :]
        distance = torch.norm(start_point - end_point, dim=1)
        distance = distance.cpu().detach().numpy()
        plt.hist(distance, bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.yscale('log')
        plt.title('Histogram of Distance Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join("histogram", "distance", f'{sub_dir}.png'))  # 保存为当前目录下的 histogram.png
        plt.close()  # 关闭绘图


    except Exception as e:
        print(f"[Warning] Failed to load data for {sub_dir}: {e}")
        continue

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

df = df.sort_values(by='dir', ascending=True)

# 打印表格
print(df.to_string(index=False))