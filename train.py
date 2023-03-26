from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()

print("Using {} dataset".format(args.dataset_str))
adj, features = load_data(args.dataset_str)
n_nodes, feat_dim = features.shape

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
test_edges = test_edges[0:100]
test_edges_false = test_edges_false[0:100]

kk = 1
zo = torch.zeros([n_nodes,kk])
feat_dim_0 = feat_dim
feat_dim = feat_dim + kk

adj = adj_train
# Some preprocessing
adj_norm = preprocess_graph(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)
adj_label = torch.FloatTensor(adj_label.toarray())
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gae_zo_label(args):
    print('===========GAE with labeling============')
    e_count = 0

    preds = []
    pos = []
    for e in test_edges:
        e_count = e_count + 1
        
        zo = torch.zeros([n_nodes,kk])
        zo[e[0],:] = 1
        zo[e[1],:] = 1
        features_test = torch.tensor(np.concatenate((np.array(features), zo), axis=1))
        
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        hidden_emb = None
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features_test, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                mu=mu, logvar=logvar, n_nodes=n_nodes,
                                norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            
            hidden_emb = mu.data.numpy()
            pre_conf = sigmoid(np.dot(hidden_emb[e[0]], hidden_emb[e[1]].T))
            print("Edge", '%04d' % (e_count), "Epoch:", '%04d' % (epoch + 1), "confidence=", "{:.5f}".format(pre_conf), "time=", "{:.5f}".format(time.time() - t))

        
        preds.append(pre_conf)
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in test_edges_false:
        e_count = e_count + 1
        
        zo = torch.zeros([n_nodes,kk])
        zo[e[0],:] = 1
        zo[e[1],:] = 1
        features_test = torch.tensor(np.concatenate((np.array(features), zo), axis=1))
        
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        hidden_emb = None
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features_test, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                mu=mu, logvar=logvar, n_nodes=n_nodes,
                                norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            
            hidden_emb = mu.data.numpy()
            pre_conf = sigmoid(np.dot(hidden_emb[e[0]], hidden_emb[e[1]].T))
            print("Edge", '%04d' % (e_count), "Epoch:", '%04d' % (epoch + 1), "confidence=", "{:.5f}".format(pre_conf), "time=", "{:.5f}".format(time.time() - t))
        
        preds_neg.append(pre_conf)
        neg.append(adj_orig[e[0], e[1]])


    # print('real')
    # print(preds)
    # print('false')
    # print(preds_neg)
    # print('================')
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))
    return roc_score, ap_score

def gae_orig(args):
    print('===========GAE without labeling=========')
    model = GCNModelVAE(feat_dim_0, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    # print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))
    return roc_score, ap_score


if __name__ == '__main__':
    roc_score_1, ap_score_1 = gae_zo_label(args)
    roc_score_2, ap_score_2 = gae_orig(args)
    print('===========GAE with labeling============')
    print('Test ROC score: ' + str(roc_score_1))
    print('Test AP score: ' + str(ap_score_1))
    print('===========GAE without labeling=========')
    print('Test ROC score: ' + str(roc_score_2))
    print('Test AP score: ' + str(ap_score_2))
