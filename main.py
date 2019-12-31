import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

import utils
import loadNCI
from model import GINClassifier


MODEL_PATH = './dump/model/'


def parse_arg():
    parser = argparse.ArgumentParser(description='GIN')
    parser.add_argument('--num-epochs', type=int, default=3000,
                        help="number of training epochs (default: 1000)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="sizes of training batches (default: 64)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for splitting the dataset into 10 (default: 0)")
    parser.add_argument('--num_layers', type=int, default=7,
                        help="number of layers INCLUDING the input one (default: 5)")
    parser.add_argument('--num_mlp_layers', type=int, default=3,
                        help="number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.")
    parser.add_argument('--neigh-pooling-type', type=str, default="sum", choices=["sum", "mean", "max"],
                        help="Pooling for over neighboring nodes: sum, mean")
    parser.add_argument('--graph-pooling-type', type=str, default="sum", choices=["sum", "mean"],
                        help="Pooling for the graph: max, sum, mean")
    parser.add_argument('--num-tasks', type=int, default=1,
                        help="number of the  task for the framework")
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help="number of hidden units")
    parser.add_argument('--feat-drop', type=float, default=0.05,
                        help="dropout rate of the feature")
    parser.add_argument('--final-drop', type=float, default=0.05,
                        help="dropout rate of the prediction layer")
    parser.add_argument('--learn-eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes.')
    args = parser.parse_args()
    return args


def train(args, nci_id):
    # fixed seed
    torch.manual_seed(0)
    # np.random.seed(0)

    # cuda test
    if torch.cuda.is_available():
        is_cuda = True
        torch.cuda.manual_seed_all(0)
    # is_cuda = False

    # load dataset
    train_set, test_set = loadNCI.load_nci_data(nci_id, cuda=is_cuda)

    # Use Pytorch's DataLoader and th collate function
    train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=loadNCI.collate)
    test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=loadNCI.collate)

    input_dim = train_set[0][0].ndata['feature'].size(1)
    output_dim = 2
    # create model
    model = GINClassifier(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.feat_drop,
                          args.learn_eps, args.graph_pooling_type, args.neigh_pooling_type, args.final_drop, is_cuda)
    if is_cuda:
        model.cuda()
    loss_func = nn.CrossEntropyLoss()  # define loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        train_acc = 0
        for iter, (bg, label) in enumerate(train_data_loader):  # bg means batch of graph
            prediction = model(bg)
            if is_cuda:
                label = label.cuda()
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            train_acc += utils.accuracy(prediction, label)
        epoch_loss /= (iter + 1)
        train_acc /= (iter + 1)

        test_acc = utils.val(model, test_data_loader, is_cuda)
        print('Task {}: Epoch{}, loss {:.4f}, TrainACC {:.4f}, TestACC {:.4f}'.format(nci_id, epoch, epoch_loss,
                                                                                      train_acc, test_acc))
        # print('Task {}: Epoch{}, loss {:.4f}, TrainACC {:.4f}'.format(nci_id, epoch, epoch_loss, train_acc))
        epoch_losses.append(epoch_loss)
    torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI-' + str(nci_id) + '.kpl')
    test_acc = utils.val(model, test_data_loader, is_cuda)
    train_acc = utils.val(model, train_data_loader, is_cuda)

    return train_acc, test_acc


def main():

    # parameters
    args = parse_arg()
    tasks_id_list = [1, 109, 123, 145, 33, 41, 47, 81, 83]
    result_list = list()

    for t in range(args.num_tasks):
        train_acc, test_acc = train(args, tasks_id_list[t])
        result_list.append([train_acc,  test_acc])

    print("\nResult of Each Task:")
    for t in range(args.num_tasks):
        print("Task {}: Train ACC: {:.4f}, Test ACC {:.4f}".format(t, result_list[t][0], result_list[t][1]))


if __name__ == '__main__':
    main()
