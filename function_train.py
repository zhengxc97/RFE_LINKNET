from model import RFE_LINKNET
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from torch.utils .data import DataLoader
import os
import argparse

def train(img_dir,label_dir,pth_path,total_epoch,batchsize,lr):

    solver = MyFrame(RFE_LINKNET, dice_bce_loss,lr)

    data=ImageFolder(img_dir,label_dir )
    dataset=DataLoader(data,batch_size= batchsize,shuffle= True ,num_workers= 2)

    no_optim = 0

    train_epoch_best_loss = 100.

    pth_out_path=os.path.join(pth_path ,'best_model.th')

    for i in range(1,total_epoch +1):
        data_loader_iter = iter(dataset)
        train_epoch_loss = 0

        for image,label in data_loader_iter :
            solver .set_input(image,label)
            loss=solver .optimize()
            train_epoch_loss +=loss
        avg_train_loss=train_epoch_loss /len(data_loader_iter)

        if avg_train_loss >=train_epoch_best_loss :
            no_optim +=1
        else:
            no_optim =0
            solver.save(pth_out_path)

        if no_optim > 6:

            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(pth_out_path)
            solver.update_lr(5.0, factor = True)


if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser(description='Training network ',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300, help='Number of epochs',
                            dest='epochs')
        parser.add_argument('-b', '--batch-size', metavar='B', default=4, help='Batch Size', dest='batchsize')
        parser.add_argument('-lr', '--learning-rate', metavar='LR', default=0.01, help='Learning Rate', dest='lr')
        parser.add_argument('-i', '--image_dir', metavar='path',type=str, default="", help='number of class', dest='numclass')
        parser.add_argument('-i', '--image_dir', metavar='path', type=str, default="",help='path of training data',
                            dest='img_dir')
        parser.add_argument('-i', '--label_dir', metavar='path', type=str, default="", help='path of training label',
                            dest='label_dir')
        parser.add_argument('-p', '--pth_dir', metavar='path', type=str, default="", help='path of result',                          dest='path of training data')
        return parser.parse_args()

arg=get_args()
train(arg.img_dir,arg.label_dir,arg.pth_dir,arg.epochs,arg.batchsize,arg.lr)