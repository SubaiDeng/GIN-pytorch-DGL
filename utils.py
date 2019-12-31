import torch
import torch.nn as nn


def knowledgeBaseInitial(in_dim, out_dim, num_layers):
    know_list = list()
    for i in range(num_layers):
        t = torch.tensor(torch.randn(in_dim, out_dim))
        know_list.append(t)
    return know_list


def accuracy(prediction, labels):
    _, indices = torch.max(prediction, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def val(model, data_loader, is_cuda):
    model.eval()
    acc = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        if is_cuda:
            label = label.cuda()
        acc += accuracy(prediction, label)
    acc /= (iter+1)
    return acc