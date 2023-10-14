import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

@torch.no_grad()
def test_gin(args, model, device, graphs, epoch):
    model.eval()
    output = pass_data_iteratively(model, graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor(
        [graph.label for graph in graphs]).to(device)
    # loss_all =  criterion_ce(output, labels)
    correct_all = pred.eq(labels.view_as(
        pred)).sum().cpu().item()
    acc_all = correct_all / len(graphs)
    mask = torch.zeros(len(graphs))
    for j in range(len(graphs)):
        mask[j] = graphs[j].nodegroup
    mask_head = (mask == 2)
    mask_medium = (mask == 1)
    mask_tail = (mask == 0)
    # loss_head =  criterion_ce(output[mask_head], labels[mask_head])
    correct_head = pred[mask_head].eq(labels[mask_head].view_as(
        pred[mask_head])).sum().cpu().item()
    acc_head = correct_head / float(mask_head.sum())
    # loss_medium =  criterion_ce(output[mask_medium], labels[mask_medium])
    correct_medium = pred[mask_medium].eq(labels[mask_medium].view_as(
        pred[mask_medium])).sum().cpu().item()
    acc_medium = correct_medium / float(mask_medium.sum())
    # loss_tail =  criterion_ce(output[mask_tail], labels[mask_tail])
    correct_tail = pred[mask_tail].eq(labels[mask_tail].view_as(
        pred[mask_tail])).sum().cpu().item()
    acc_tail = correct_tail / float(mask_tail.sum())
    return acc_all, acc_head, acc_medium, acc_tail

    # print("accuracy train: %f test: %f" % (acc_train, acc_test))
    #
    # return acc_train, acc_test
def data_split(graph_list, valid_ratio=0.1, test_ratio=0.2, seed=2022):
    random.seed(seed)
    shuffled_indices = list(range(len(graph_list)))
    random.shuffle(shuffled_indices)
    test_set_size = int(len(graph_list) * test_ratio)
    train_set_size = int(len(graph_list) * (1 - test_ratio - valid_ratio))
    test_indices = shuffled_indices[-test_set_size:]
    valid_indices = shuffled_indices[train_set_size:-test_set_size]
    train_indices = shuffled_indices[:train_set_size]
    train_graph_list = [graph_list[i] for i in train_indices]
    # train_sample_list = [sample_list[i] for i in train_indices]
    test_graph_list = [graph_list[i] for i in test_indices]
    valid_graph_list = [graph_list[i] for i in valid_indices]

    return train_graph_list, valid_graph_list, test_graph_list

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    if args.dataset == "PROTEINS":
        hidden_dim = 32
        batch_size = 32
        seed = 2022
        learn_eps = False
        l2 = 0
        K = [0, 371, 742, 1113]
    elif args.dataset == "PTC":
        hidden_dim = 32
        batch_size = 32
        seed = 0
        learn_eps = True
        l2 = 5e-4
        K = [0, 115, 230, 344]
    elif args.dataset == "IMDBBINARY":
        hidden_dim = 64
        batch_size = 32
        seed = 2020
        learn_eps = True
        l2 = 5e-4
        K = [0, 333, 666, 1000]
    elif args.dataset == "DD":
        hidden_dim = 32
        batch_size = 128
        seed = 2022
        learn_eps = False
        l2 = 0
        K = [0, 393, 785, 1178]
    elif args.dataset == "FRANK":
        hidden_dim = 32
        batch_size = 128
        seed = 2022
        learn_eps = True
        l2 = 5e-4
        K =[0, 1445, 2890, 4337]
    else:
        K = [0, 1370, 2740, 4110]


    nodes = torch.zeros(len(graphs))

    for i in range(len(graphs)):
        nodes[i] = graphs[i].g.number_of_nodes()

    _, ind = torch.sort(nodes, descending=True)

    for i in ind[K[0]:K[1]]:
        graphs[i].nodegroup = 2
    for i in ind[K[1]:K[2]]:
        graphs[i].nodegroup = 1
    for i in ind[K[2]:K[3]]:
        graphs[i].nodegroup = 0

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, valid_graphs, test_graphs = data_split(graphs, seed=args.seed)


    times = 5

    test_record = torch.zeros(times)
    valid_record = torch.zeros(times)
    head_record = torch.zeros(times)
    medium_record = torch.zeros(times)
    tail_record = torch.zeros(times)

    for seed in range(times):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        best_valid_acc = 0

        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
            # acc_train, acc_test = test(args, model, device, train_graphs, valid_graphs, epoch)
            acc_valid, acc_head_valid, acc_medium_valid, acc_tail_valid = test_gin(args, model, device,
                                                                                   valid_graphs, epoch)

            # print("valid loss: %.4f acc: %.4f" % (loss_valid, acc_valid))

            if acc_valid > best_valid_acc:
                best_valid_acc = acc_valid
                # best_valid_loss = loss_valid
                patience = 0
                loss, test_acc, test_acc_head, test_acc_medium, test_acc_tail = test_gin(args, model, device, test_graphs,
                                                                                         epoch)
                print("test acc: %.4f" % test_acc)
                print("test acc_head: %.4f" % test_acc_head)
                print("test acc_medium: %.4f" % test_acc_medium)
                print("test acc_tail: %.4f" % test_acc_tail)
                # test_acc_list.append(test_acc)
                # test_acc_head_list.append(test_acc_head)
                # test_acc_medium_list.append(test_acc_medium)
                # test_acc_tail_list.append(test_acc_tail)

            # if not args.filename == "":
            #     with open(args.filename, 'w') as f:
            #         f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
            #         f.write("\n")
            # print("")

        test_record[seed] = test_acc
        valid_record[seed] = best_valid_acc
        head_record[seed] = test_acc_head
        medium_record[seed] = test_acc_medium
        tail_record[seed] = test_acc_tail

    print('Valid mean: %.4f, std: %.4f' %
          (valid_record.mean().item(), valid_record.std().item()))
    print('Test mean: %.4f, std: %.4f' %
          (test_record.mean().item(), test_record.std().item()))
    print('Head mean: %.4f, std: %.4f' %
          (head_record.mean().item(), head_record.std().item()))
    print('Medium mean: %.4f, std: %.4f' %
          (medium_record.mean().item(), medium_record.std().item()))
    print('Tail mean: %.4f, std: %.4f' %
          (tail_record.mean().item(), tail_record.std().item()))



if __name__ == '__main__':
    main()
