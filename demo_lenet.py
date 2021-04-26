import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class BitLoser(nn.Module):
    def __init__(self, approx_bits, loss_prob, num_refreshes):
        super(BitLoser, self).__init__()
        self.approx_bits = approx_bits
        self.loss_prob = 1 - ((1 - loss_prob) ** max(1, num_refreshes))
        print('ploss = {}'.format(self.loss_prob))
        return

    def forward(self, x):
        if self.training:
            return x
        else:
            mask = np.zeros(x.shape, dtype=np.int32)
            for b in self.approx_bits:
                l = np.random.random(x.shape) < self.loss_prob
                mask = mask | np.left_shift(l.astype(np.int32), b)
            y = torch.from_numpy((x.numpy().view(np.int32) & np.invert(mask))\
                                    .view(np.float32))
            return y

class LeNet(nn.Module):
    def __init__(self, approx=False):
        super(LeNet, self).__init__()
        self.approx = approx
        NUM_APPROX_BITS = 23
        P_LOSS1 = 1e-1
        NUM_REFRESHES = 10
        if approx:
            self.bl = BitLoser(range(NUM_APPROX_BITS), P_LOSS1, NUM_REFRESHES)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.conv3 = nn.Conv2d(16, 120, 5, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        return

    def forward(self, X):
        if self.approx:
            X = self.bl(X)
        y = self.conv1(X)
        if self.approx:
            y = self.bl(y)
        y = F.relu(y)
        if self.approx:
            y = self.bl(y)
        y = F.max_pool2d(y, 2)
        if self.approx:
            y = self.bl(y)
        y = self.conv2(y)
        if self.approx:
            y = self.bl(y)
        y = F.relu(y)
        if self.approx:
            y = self.bl(y)
        y = F.max_pool2d(y, 2)
        if self.approx:
            y = self.bl(y)
        y = self.conv3(y)
        if self.approx:
            y = self.bl(y)
        y = F.relu(y)
        if self.approx:
            y = self.bl(y)
        y = torch.flatten(y, 1)
        if self.approx:
            y = self.bl(y)
        y = self.fc1(y)
        if self.approx:
            y = self.bl(y)
        y = F.relu(y)
        if self.approx:
            y = self.bl(y)
        y = self.fc2(y)
        if self.approx:
            y = self.bl(y)
        y = F.log_softmax(y, dim=1)
        if self.approx:
            y = self.bl(y)
        return y

def train(loader, model, optimizer):
    # stolen from https://github.com/pytorch/examples/blob/master/mnist/main.py
    model.train()
    for X, y_true in loader:
        optimizer.zero_grad()
        y_out = model(X)
        loss = F.nll_loss(y_out, y_true)
        loss.backward()
        optimizer.step()
    return

def test(loader, model):
    # also stolen from
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y_true in loader:
            y_out = model(X)
            test_loss += F.nll_loss(y_out, y_true, reduction='sum').item()
            y_pred = y_out.argmax(dim=1, keepdim=True)
            correct += y_pred.eq(y_true.view_as(y_pred)).sum().item()
    test_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('test_loss: {:.4f}, test_acc: {:.4f}'.format(test_loss, acc))
    return

def main():
    torch.set_default_dtype(torch.float32)
    TEST_BATCH_SIZE = 10000
    tform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    test_data = datasets.MNIST('./data', train=False, download=True,
                                transform=tform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=TEST_BATCH_SIZE)
    APPROX = True
    model = LeNet(approx=APPROX).to('cpu')
    SAVED_MODEL_PARAMS_FILE = './lenet_mnist.pt'
    if os.path.exists(SAVED_MODEL_PARAMS_FILE):
        # load saved params
        ckpt = torch.load(SAVED_MODEL_PARAMS_FILE)
        model.load_state_dict(ckpt, strict=True)
    else:
        # train model
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=tform)
        TRAIN_BATCH_SIZE = 1000
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=TRAIN_BATCH_SIZE)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=0.7)
        EPOCHS = 10
        for epoch in range(1, EPOCHS+1):
            test(test_loader, model)
            print('Epoch {}'.format(epoch))
            train(train_loader, model, optimizer)
            sched.step()
        torch.save(model.state_dict(), SAVED_MODEL_PARAMS_FILE)
    test(test_loader, model)
    return

if __name__ == '__main__':
    main()

