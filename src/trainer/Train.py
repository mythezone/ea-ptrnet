"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""
import comet_ml
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm



# Insert the path into sys.path for importing.
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.PointerNet import PointerNet
from data.data_generator import TSPDataset

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

exp = comet_ml.Experiment(project_name="ptea", workspace="mythezone")

#region Oringinal Parameters         
########################################################

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')

parser.add_argument('--file_path',default="./dataset/train/exp_default.pth" ,type=str, help='Data file path')

#######################################################
# endregion



# region Test Parameters
#######################################################
# Data
# parser.add_argument('--train_size', default=1024, type=int, help='Training data size')
# parser.add_argument('--val_size', default=128, type=int, help='Validation data size')
# parser.add_argument('--test_size', default=16, type=int, help='Test data size')
# parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# # Train
# parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# # GPU
# parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# # TSP
# parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# # Network
# parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
# parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
# parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
# parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
# parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')
# parser.add_argument('--file_path',default="./dataset/train/test_exp.pth" ,type=str, help='Data file path')

# endregion 


params = parser.parse_args()

exp.log_parameters(vars(params))

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

data_file = params.file_path

dataset = TSPDataset(params.train_size,
                     params.nof_points,
                     file_path=data_file)

dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=4)

if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []



for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        losses.append(loss.item())
        batch_loss.append(loss.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()
        if i_batch % 100 == 0:
            exp.log_metric("batch_loss", loss.item(), step=i_batch+epoch*len(dataloader))

        iterator.set_postfix(loss='{}'.format(loss.item()))
        
    # exp.log_metric("epoch_loss", np.average(batch_loss), step=epoch)
    
    iterator.set_postfix(loss=np.average(batch_loss))
    
exp.end()
