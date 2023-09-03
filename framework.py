import torch
import torch.nn as nn
from torch.autograd import Variable as V
import os
import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net().to(device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
    #读取输入以及图片id
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id


    #预测一张影像
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    #批量预测图像
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id

    #根据图片位置预测一张影像
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask

    #将图片以及标签赚到cuda
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    #模型优化
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data.item()
    #保存模型
    def save(self, path):
        torch.save(self.net.state_dict(), path)
    #读取模型
    def load(self, path):
        # checkpoint = torch.load(path)
        # ckpt = checkpoint['state_dict']
        # newcheckpoint = {}
        # for k, v in ckpt.items():
        #     k =  k[7:]
        #     newcheckpoint[k] = v
        self.net.load_state_dict(torch.load(path))
    #更新学习率
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.old_lr = new_lr
