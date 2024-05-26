import os
import sys
import json
import pickle
import random
import math

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


import shutil

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        train_folder = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        for folder in train_folder:
            train_images = [os.path.join(folder, j) for j in os.listdir(folder)
                            if os.path.splitext(j)[-1] in supported]
            # 排序，保证各平台顺序一致
            train_images.sort()
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(train_images))
            # # 按比例随机采样验证样本
            # for img_path in train_images:
            #     train_images_path.append(img_path)
            #     train_images_label.append(image_class)
            # 按比例随机采样验证样本
            train_images_path.append(train_images)
            train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

    return train_images_path, train_images_label


def read_split_data_test(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [[os.path.join(root, cla, i)] for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
    # for cla in flower_class:
    #     cla_path = os.path.join(root, cla)
    #     # 遍历获取supported支持的所有文件路径
    #     train_folder = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
    #     for folder in train_folder:
    #         train_images = [os.path.join(folder, j) for j in os.listdir(folder)
    #                         if os.path.splitext(j)[-1] in supported]
    #         # 排序，保证各平台顺序一致
    #         train_images.sort()
    #         # 获取该类别对应的索引
    #         image_class = class_indices[cla]
    #         # 记录该类别的样本数量
    #         every_class_num.append(len(train_images))
            # # 按比例随机采样验证样本
            # for img_path in train_images:
            #     train_images_path.append(img_path)
            #     train_images_label.append(image_class)
        # 按比例随机采样验证样本
        for i in images:
            train_images_path.append(i)
            train_images_label.append(image_class)
        # train_images_path.append(images)
        # train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

    return train_images_path, train_images_label

# def read_split_data(root: str, val_rate: float = 0.2):
#     random.seed(0)  # 保证随机结果可复现
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
#
#     # 遍历文件夹，一个文件夹对应一个类别
#     flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
#     # 排序，保证各平台顺序一致
#     flower_class.sort()
#     # 生成类别名称以及对应的数字索引
#     class_indices = dict((k, v) for v, k in enumerate(flower_class))
#     json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     train_images_path = []  # 存储训练集的所有图片路径
#     train_images_label = []  # 存储训练集图片对应索引信息
#     val_images_path = []  # 存储验证集的所有图片路径
#     val_images_label = []  # 存储验证集图片对应索引信息
#     every_class_num = []  # 存储每个类别的样本总数
#     supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
#     # 遍历每个文件夹下的文件
#     for cla in flower_class:
#         cla_path = os.path.join(root, cla)
#         # 遍历获取supported支持的所有文件路径
#         images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
#                   if os.path.splitext(i)[-1] in supported]
#         # 排序，保证各平台顺序一致
#         images.sort()
#         # 获取该类别对应的索引
#         image_class = class_indices[cla]
#         # 记录该类别的样本数量
#         every_class_num.append(len(images))
#         # 按比例随机采样验证样本
#         val_path = random.sample(images, k=int(len(images) * val_rate))
#
#         for img_path in images:
#             if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
#                 val_images_path.append(img_path)
#                 val_images_label.append(image_class)
#             else:  # 否则存入训练集
#                 train_images_path.append(img_path)
#                 train_images_label.append(image_class)
#
#     print("{} images were found in the dataset.".format(sum(every_class_num)))
#     print("{} images for training.".format(len(train_images_path)))
#     print("{} images for validation.".format(len(val_images_path)))
#     # assert len(train_images_path) > 0, "number of training images must greater than 0."
#     # assert len(val_images_path) > 0, "number of validation images must greater than 0."
#
#     plot_image = False
#     if plot_image:
#         # 绘制每种类别个数柱状图
#         plt.bar(range(len(flower_class)), every_class_num, align='center')
#         # 将横坐标0,1,2,3,4替换为相应的类别名称
#         plt.xticks(range(len(flower_class)), flower_class)
#         # 在柱状图上添加数值标签
#         for i, v in enumerate(every_class_num):
#             plt.text(x=i, y=v + 5, s=str(v), ha='center')
#         # 设置x坐标
#         plt.xlabel('image class')
#         # 设置y坐标
#         plt.ylabel('number of images')
#         # 设置柱状图的标题
#         plt.title('flower class distribution')
#         plt.show()
#
#     return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

class Myloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels, device="cuda:0"):
        num = torch.tensor([0.]).long().to(device)
        p = torch.zeros(pred.size(0)).to(device)
        all_p = torch.softmax(pred, dim=1).to(device)
        max_values, _ = torch.max(all_p, dim=1) # 沿着维度1（每行）计算最大值
        sorted_values, _ = torch.sort(all_p, dim=1, descending=True) # 沿着维度1（每行）进行降序排序
        second_max_values = sorted_values[:, 1] # 取每行的第二个元素（第二大值）
        diff_values = max_values - second_max_values
        # 找到所有差值中的最小值
        min_diff = torch.min(diff_values)
        y = 3 * diff_values
        y = y.to(device)
        for i in labels:
            right_p = all_p[num[0]][i]
            p[num[0]] = right_p
            num[0] += 1
        loss = -torch.mean((torch.pow((torch.ones(pred.size(0)).to(device)-p), y))*torch.log(p))
        return loss

# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
#     model.train()
#     loss_function = torch.nn.CrossEntropyLoss()
#     # loss_function = Myloss()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     optimizer.zero_grad()
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels, _ = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = loss_function(pred, labels.to(device))
#         loss.backward()
#         accu_loss += loss.detach()
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             optimizer.param_groups[0]["lr"]
#         )
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         optimizer.step()
#         optimizer.zero_grad()
#         # update lr
#         lr_scheduler.step()
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
def train_one_epoch_dtfd(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model[0].train()
    model[1].train()
    model[2].train()
    model[3].train()

    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        images = images.squeeze(0) #B,N,C,W,H -> N,C,W,H
        slide_pseudo_feat = [] #用于存放
        slide_sub_feat = []
        slide_sub_labels = []

        for i in range(len(images)//10): #N,C,W,H -> N,C,W,H
            feat = model[2](images[10*i:10*(i+1)].unsqueeze(0).to(device)).squeeze(0) #N,C,W,H -> N,512
            tAA = model[1](feat).squeeze(0) # N,512 -> 1,N -> N
            feat = torch.sum(torch.einsum('ns,n->ns', feat, tAA), dim=0).unsqueeze(0) #1,512
            tpred = model[0](feat) #1,classes_num
            slide_sub_feat.append(tpred)
            slide_sub_labels.append(labels)

            slide_pseudo_feat.append(feat)

        #first optimization
        slide_sub_feat = torch.cat(slide_sub_feat, dim=0)
        # slide_sub_feat = torch.max(slide_sub_feat, dim=1)[1]
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        pred = model[3](slide_pseudo_feat)
        pred_classes = torch.max(pred, dim=1)[1]
        loss1 = loss_function(pred, labels.to(device))
        loss0 = loss_function(slide_sub_feat, slide_sub_labels.to(device))
        loss = loss1 + loss0
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model[0].parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model[1].parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model[2].parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model[3].parameters(), 5)
        optimizer.step()

        # # first optimization
        # slide_sub_feat = torch.cat(slide_sub_feat, dim=0)
        # # slide_sub_feat = torch.max(slide_sub_feat, dim=1)[1]
        # slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
        # loss0 = loss_function(slide_sub_feat, slide_sub_labels.to(device))
        # optimizer.zero_grad()
        # loss0.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model[0].parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(model[1].parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(model[2].parameters(), 5)
        # optimizer.step()
        #
        # # second optimization
        # slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        # pred = model[3](slide_pseudo_feat)
        # pred_classes = torch.max(pred, dim=1)[1]
        # loss1 = loss_function(pred, labels.to(device))
        # optimizer.zero_grad()
        # loss1.backward()
        # torch.nn.utils.clip_grad_norm_(model[3].parameters(), 5)
        # optimizer.step()

            # pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_dtfd(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()

    model[0].eval()
    model[1].eval()
    model[2].eval()
    model[3].eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    pred_probs = []
    true_labels = []
    pred_cls = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        images = images.squeeze(0)  # B,N,C,W,H -> N,C,W,H
        slide_pseudo_feat = []  # 用于存放

        for i in range(len(images)//10): #N,C,W,H -> N,C,W,H
            feat = model[2](images[10*i:10*(i+1)].unsqueeze(0).to(device)).squeeze(0) #N,C,W,H -> N,512
            tAA = model[1](feat).squeeze(0) # N,512 -> 1,N -> N
            feat = torch.sum(torch.einsum('ns,n->ns', feat, tAA), dim=0).unsqueeze(0) #1,512

            slide_pseudo_feat.append(feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        pred = model[3](slide_pseudo_feat)
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 计算 AUC 所需的预测概率和真实标签
        pred_probs.extend(torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu().detach().numpy())
        true_labels.extend(labels.cpu().detach().numpy())
        pred_cls.extend(torch.max(pred, dim=1)[1].cpu().detach().numpy())  # 获取预测的类别

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 计算 AUC
    auc = roc_auc_score(true_labels, pred_probs)
    precision = precision_score(true_labels, pred_cls, zero_division=0)
    recall = recall_score(true_labels, pred_cls, zero_division=0)
    f1 = f1_score(true_labels, pred_cls, zero_division=0)

    combined = list(zip(pred_probs, true_labels, pred_cls))
    import csv
    filename = r'csv\私人_垂体腺瘤\dtfd' + str(epoch) + '.csv'
    with open(filename, 'w', newline='') as file:
        # 创建一个csv.writer对象
        writer = csv.writer(file)

        # 写入数据
        for row in combined:
            writer.writerow(row)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    pred_probs = []
    true_labels = []
    pred_cls = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 计算 AUC 所需的预测概率和真实标签
        pred_probs.extend(torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu().detach().numpy())
        true_labels.extend(labels.cpu().detach().numpy())
        pred_cls.extend(torch.max(pred, dim=1)[1].cpu().detach().numpy())  # 获取预测的类别

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 计算 AUC
    auc = roc_auc_score(true_labels, pred_probs)
    precision = precision_score(true_labels, pred_cls, zero_division=0)
    recall = recall_score(true_labels, pred_cls, zero_division=0)
    f1 = f1_score(true_labels, pred_cls, zero_division=0)

    print("best_acc =", accu_num.item() / sample_num, "auc =", auc, "precision =", precision, "recall =", recall, "f1 =", f1)

    combined = list(zip(pred_probs, true_labels, pred_cls))
    import csv
    filename = r'csv\私人_垂体腺瘤\new\Embedding+max#' + str(epoch) + '.csv'
    with open(filename, 'w', newline='') as file:
        # 创建一个csv.writer对象
        writer = csv.writer(file)

        # 写入数据
        for row in combined:
            writer.writerow(row)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1

def train_one_our_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, feature_extractor):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        images = feature_extractor(images.to(device))
        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_our(model, data_loader, device, epoch, feature_extractor):
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    pred_probs = []
    true_labels = []
    pred_cls = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        images = feature_extractor(images.to(device))
        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # print(torch.max(torch.nn.functional.softmax(pred, dim=1), dim=1)[0])

        # 计算 AUC 所需的预测概率和真实标签
        pred_probs.extend(torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu().detach().numpy())
        true_labels.extend(labels.cpu().detach().numpy())
        pred_cls.extend(torch.max(pred, dim=1)[1].cpu().detach().numpy())  # 获取预测的类别

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 计算 AUC
    auc = roc_auc_score(true_labels, pred_probs)
    precision = precision_score(true_labels, pred_cls, zero_division=0)
    recall = recall_score(true_labels, pred_cls, zero_division=0)
    f1 = f1_score(true_labels, pred_cls, zero_division=0)

    print("best_acc =", accu_num.item() / sample_num, "auc =", auc, "precision =", precision, "recall =", recall, "f1 =", f1)

    combined = list(zip(pred_probs, true_labels, pred_cls))
    import csv
    filename = r'csv\公开\MSSFNet4#' + str(epoch) + '.csv'
    with open(filename, 'w', newline='') as file:
        # 创建一个csv.writer对象
        writer = csv.writer(file)

        # 写入数据
        for row in combined:
            writer.writerow(row)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1

# @torch.no_grad()
# def evaluate_our(model, data_loader, device, epoch, feature_extractor):
#     loss_function = torch.nn.CrossEntropyLoss()
#     # loss_function = Myloss()
#
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     pred_probs = []
#     true_labels = []
#     pred_cls = []
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#
#         images = feature_extractor(images.to(device))
#         pred = model(images)
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#         # print(torch.max(torch.nn.functional.softmax(pred, dim=1), dim=1)[0])
#
#         # 计算 AUC 所需的预测概率和真实标签
#         pred_cls.append(torch.max(pred, dim=1)[1].cpu().detach().item())  # 获取预测的类别
#         pred_probs.append(torch.nn.functional.softmax(pred, dim=1)[:, pred_classes].cpu().detach().item())
#         true_labels.extend(labels.cpu().detach().numpy())
#
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num
#         )
#
#     # 计算 AUC
#     import csv
#     combined_data = [pred_cls, pred_probs]
#     csv_file_name = '1027.csv'
#     with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
#         # 创建一个csv写入器
#         writer = csv.writer(file)
#
#         # 遍历combined_data列表，将每一行数据写入CSV
#         for data_row in combined_data:
#             # 将列表转换为字符串，以便写入CSV
#             writer.writerow(data_row)
#
#     auc = roc_auc_score(true_labels, pred_probs)
#     precision = precision_score(true_labels, pred_cls, zero_division=0)
#     recall = recall_score(true_labels, pred_cls, zero_division=0)
#     f1 = f1_score(true_labels, pred_cls, zero_division=0)
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1

@torch.no_grad()
def evaluate_our_test(model, data_loader, device, epoch, feature_extractor):
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    pred_probs = []
    true_labels = []
    pred_cls = []

    copy_root = r"C:\Users\LinYi\baidudownload\lunwen_新筛选\val/"
    num = 0

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, path = data
        sample_num += images.shape[0]
        path = path[0]

        images = feature_extractor(images.to(device))
        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        pred_ = torch.max(torch.nn.functional.softmax(pred, dim=1), dim=1)[0].item()
        if pred_ >= 0.7:
            num += 1
            print(num)
            print(pred_)
            name = path.split("\\")[-1].split(".")[0]
            Cla = path.split("\\")[-2]
            shutil.copyfile(path, os.path.join(copy_root, Cla, name + ".jpg"))


        # 计算 AUC 所需的预测概率和真实标签
        pred_probs.extend(torch.nn.functional.softmax(pred, dim=1)[:, 1].cpu().detach().numpy())
        true_labels.extend(labels.cpu().detach().numpy())
        pred_cls.extend(torch.max(pred, dim=1)[1].cpu().detach().numpy())  # 获取预测的类别

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 计算 AUC
    auc = roc_auc_score(true_labels, pred_probs)
    precision = precision_score(true_labels, pred_cls, zero_division=0)
    recall = recall_score(true_labels, pred_cls, zero_division=0)
    f1 = f1_score(true_labels, pred_cls, zero_division=0)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1

# def evaluate(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()
#     # loss_function = Myloss()
#
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     last_name = 0
#     patch_num = 0
#     patch_acc = 0
#     wsi_num = 0
#     wsi_acc = 0
#     for step, data in enumerate(data_loader):
#         images, labels, path = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         for i in range(images.shape[0]):
#             name = path[i].split("\\")[-2]
#             if name == last_name:
#                 patch_num += 1
#                 if torch.eq(pred_classes[i], labels[i].to(device)):
#                     patch_acc += 1
#             else:
#                 wsi_num += 1
#                 if patch_acc*2 > patch_num:
#                     wsi_acc += 1
#                 last_name = name
#                 patch_num = 0
#                 patch_acc = 0
#
#
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, wsi_acc: {:.3f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             wsi_acc / wsi_num
#         )
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, wsi_acc / wsi_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
