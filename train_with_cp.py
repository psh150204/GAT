import os 
import time 
import random 
import argparse
import glob

import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable

from utils import accuracy, load_data, save_checkpoint
from model import GCN, GAT, SpGCN, SpGAT
 
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if(torch.cuda.is_available()) else 'cpu')

    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    N = features.size(0) # num_nodes
    C = features.size(1) # num_of_features
    H = args.hidden # hidden nodes
    F = labels.max().item() + 1 # num_classes
    
    epochs_since_improvement = 0
    best_acc = 0.

    network = GCN(C, H, F, args.dropout).to(device)
    optimizer = optim.Adam(network.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # pre-processing
    A_tilde = adj + torch.eye(N)
    D_tilde_inv_sqrt = torch.diag(torch.sqrt(torch.sum(A_tilde, dim = 1)) ** -1)
    A_hat = torch.mm(D_tilde_inv_sqrt, torch.mm(A_tilde, D_tilde_inv_sqrt)).to(device)
    
    features = features.to(device)
    labels = labels.to(device)

    if not args.test :
        for epoch in range(args.epochs):
            t = time.time()
            network.train()

            preds = network(features, A_hat) # [N, F]
            train_loss = criterion(preds[idx_train], labels[idx_train])
            train_acc = accuracy(preds[idx_train], labels[idx_train])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # validation
            with torch.no_grad():
                network.eval()
                preds_val = network(features, A_hat)
                val_loss = criterion(preds_val[idx_val], labels[idx_val])
                val_acc = accuracy(preds_val[idx_val], labels[idx_val])

                test_loss = criterion(preds_val[idx_test], labels[idx_test])
                test_acc = accuracy(preds_val[idx_test], labels[idx_test])

                # early stopping
                if val_acc > best_acc :
                    best_acc = val_acc
                    epochs_since_improvement = 0
                    save_checkpoint(network, 'checkpoints/best.pt')
                else :
                    epochs_since_improvement += 1
            
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(network, 'checkpoints/epoch_%d.pt'%(epoch+1))

            print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(train_loss.item()),
            'acc_train: {:.4f}'.format(train_acc.item()),
            'loss_val: {:.4f}'.format(val_loss.item()),
            'acc_val: {:.4f}'.format(val_acc.item()),
            'loss_test: {:.4f}'.format(test_loss.item()),
            'acc_test: {:.4f}'.format(test_acc.item()),
            'time: {:.4f}s'.format(time.time() - t))

            if epochs_since_improvement > args.patience - 1 :
                print("There's no improvements during %d epochs and so stop the training."%(args.patience))
                break

        preds = network(features, A_hat)
        test_acc = accuracy(preds[idx_test], labels[idx_test])
        print('Test Accuracy : %.4f'%(test_acc))
        
    else :
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            network.load_state_dict(checkpoint['state_dict'])
            print("trained network " + args.checkpoint + " is loaded successfully")
        else:
            raise ValueError("There's no such files or directory")
        
        with torch.no_grad():
            network.eval()
            preds = network(features, A_hat)
            test_acc = accuracy(preds[idx_test], labels[idx_test])
            print('Test Accuracy : %.4f'%(test_acc))

if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False, help='Testing.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=500, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer'], help='Dataset to train.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',help='Path to the checkpoint')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)