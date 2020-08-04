import os 
import time 
import random 
import argparse
import glob
import matplotlib.pyplot as plt

import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable

from utils import accuracy, load_data
from model import GCN, GAT, SpGCN, SpGAT
 
def main(args):
    # meta settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if(torch.cuda.is_available()) else 'cpu')

    # load the data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    features = features.to(device)
    labels = labels.to(device)

    # parameter intialization
    N = features.size(0) # num_of_nodes
    F = features.size(1) # num_of_features
    H = args.hidden # hidden nodes
    C = labels.max().item() + 1 # num_classes
    
    # for validation
    epochs_since_improvement = 0
    best_loss = 10.

    # init training object
    if args.model == 'GCN':
        network = GCN(F, H, C, args.dropout).to(device)

        # pre-processing
        A_tilde = adj + torch.eye(N)
        D_tilde_inv_sqrt = torch.diag(torch.sqrt(torch.sum(A_tilde, dim = 1)) ** -1)
        adj = torch.mm(D_tilde_inv_sqrt, torch.mm(A_tilde, D_tilde_inv_sqrt)).to(device) # A_hat
        
    else:
        network = GAT(F, H, C, args.dropout, args.alpha, args.n_heads).to(device)
        adj = adj.to(device)
    
    optimizer = optim.Adam(network.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    # Training
    for epoch in range(args.epochs):
        t = time.time()
        network.train()

        preds = network(features, adj) # [N, F]
        train_loss = criterion(preds[idx_train], labels[idx_train])
        train_acc = accuracy(preds[idx_train], labels[idx_train])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # validation
        with torch.no_grad():
            network.eval()
            preds_val = network(features, adj)
            val_loss = criterion(preds_val[idx_val], labels[idx_val])
            val_acc = accuracy(preds_val[idx_val], labels[idx_val])

            # early stopping
            if val_loss < best_loss :
                best_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        train_losses.append(train_loss.item())
        train_accs.append(train_acc.item())
        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())

        print('[%d/%d] train loss : %.4f | train acc %.2f%% | val loss %.4f | val acc %.2f%% | time %.3fs'
                    %(epoch+1, args.epochs, train_loss.item(), train_acc.item() * 100, val_loss.item(), val_acc.item() * 100, time.time() - t))

        if epochs_since_improvement > args.patience - 1 :
            print("There's no improvements during %d epochs and so stop the training."%(args.patience))
            break
    
    # Testing
    with torch.no_grad():
        network.eval()
        preds = network(features, adj)
        test_acc = accuracy(preds[idx_test], labels[idx_test])
        print('Test Accuracy : %.2f'%(test_acc * 100))

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(train_losses, label = 'train loss')
    ax.plot(val_losses, label = 'validation loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('cross entropy loss')
    ax.legend()

    ax.set(title="Loss Curve of " + args.model + " on " + args.dataset)
    ax.grid()

    fig.savefig("results/"+args.model+"_"+args.dataset+"_loss_curve.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(train_accs, label = 'train accuracy')
    ax.plot(val_accs, label = 'validation accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()

    ax.set(title="Accuracy Graph of " + args.model + " on " + args.dataset + " : Test Accuracy %.4f"%(test_acc))
    ax.grid()

    fig.savefig("results/"+args.model+"_"+args.dataset+"_accuracy.png")
    plt.close()

if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=500, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer'], help='Dataset to train.')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN','GAT'], help='Model to train.')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)