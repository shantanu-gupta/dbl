import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy

class BitError(nn.Module):
    @staticmethod
    def generate_mask(shape, approx_bits, loss_prob):
        mask = np.zeros(shape, dtype=np.int32)
        if loss_prob > 0.0:
            for b in approx_bits:
                l = np.random.random(shape) < loss_prob
                mask = mask | np.left_shift(l.astype(np.int32), b)
        return mask

    @staticmethod
    def apply_mask(x, mask):
        return torch.from_numpy((x.numpy().view(np.int32) & np.invert(mask)) \
                                .view(np.float32))

class LeNet(nn.Module):
    NUM_APPROX_BITS = 23
    P_LOSS1 = 1e-1
    MIN_READ_DELAY = 30     # write-to-read delay
    
    def __init__(self, approx=False):
        super(LeNet, self).__init__()
        self.approx = approx
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 120, 5, padding=0)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(120, 84)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.lsm = nn.LogSoftmax(1)
        # hack to allow bit loss at the end
        # self.final = nn.Identity()
        self.layers = nn.Sequential(self.conv1, self.relu1, self.pool1,
                                    self.conv2, self.relu2, self.pool2,
                                    self.conv3, self.relu3, self.flatten,
                                    self.fc1, self.relu_fc1,
                                    self.fc2, self.lsm)
                                    # self.final)

    def forward(self, x):
        if not self.approx:
            x = self.layers(x)
        else:
            assert not self.training, "can't handle bitloss in training!"
            for i, layer in enumerate(self.layers):
                p_dataloss = 1 - ((1 - LeNet.P_LOSS1)
                                    ** max(1, LeNet.MIN_READ_DELAY))
                data_mask = BitError.generate_mask(x.shape,
                                                range(LeNet.NUM_APPROX_BITS),
                                                p_dataloss)
                x = BitError.apply_mask(x, data_mask)
                layer_copy = copy.deepcopy(layer)
                p_paramloss = 1 - ((1 - LeNet.P_LOSS1)
                                    ** max(1, LeNet.MIN_READ_DELAY * (i+1)))
                # p_paramloss = 0
                for t in layer_copy.parameters():
                    param_mask = BitError.generate_mask(t.data.shape,
                                                range(LeNet.NUM_APPROX_BITS),
                                                p_paramloss)
                    t.data = BitError.apply_mask(t.data, param_mask)
                x = layer_copy(x)
        return x

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

    model = LeNet().to('cpu')
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
            print('Test before epoch {}'.format(epoch))
            test(test_loader, model)
            print('Train epoch {}'.format(epoch))
            train(train_loader, model, optimizer)
            sched.step()
        print('Test at end')
        test(test_loader, model)
        torch.save(model.state_dict(), SAVED_MODEL_PARAMS_FILE)
    # now turn approx on
    print('Allowing bit-errors now...')
    model.approx = True
    test(test_loader, model)
    return

if __name__ == '__main__':
    main()

