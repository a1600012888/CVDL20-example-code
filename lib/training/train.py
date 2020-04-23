import os
import sys
father_dir = os.path.join('/',  *os.path.realpath(__file__).split(os.path.sep)[:-2])
#print(father_dir)
if not father_dir in sys.path:
    sys.path.append(father_dir)
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm

def train_one_epoch(net, batch_generator, optimizer,
                    criterion, DEVICE=torch.device('cuda:0'),
                    descrip_str='Training', **args):
    '''

    :param net: xxx

    :return:  None    #(clean_acc, adv_acc)
    '''
    net.train()
    pbar = tqdm(batch_generator)

    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        #print('data shape', data.shape, label.shape)
        optimizer.zero_grad()

        pbar_dic = OrderedDict()
        TotalLoss = 0


        pred = net(data)

        loss = criterion(pred, label)
        #TotalLoss = TotalLoss + loss
        loss.backward()

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss.item()
        #pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)

        pbar.set_postfix(pbar_dic)


def eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0')):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()


    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)


        pbar.set_postfix(pbar_dic)


    return clean_accuracy.mean
