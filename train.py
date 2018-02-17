import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchnet import meter


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def train(model, train_data, num_epochs=100,print_ever=200):
    model.apply(reset)
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        #print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        loss_meter = meter.AverageValueMeter()
        model.train()
        for x, y in train_data:
            x_var = Variable(x)
            y_var = Variable(y).view(1,-1)
            #scores = model(x_var, y_var) # 引导模式
            scores,_ = model.viterbi(x_var) # 维特比模式
            loss_ = loss_fn(scores.squeeze_(), y_var.squeeze_())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            loss_meter.add(loss_.data[0])
        if epoch % print_ever == 0:
            print("epoch:{}:loss:{}".format(epoch, loss_meter.value()[0]))
