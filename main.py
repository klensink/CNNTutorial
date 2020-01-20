
# 1) Imports
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from models.resnet import resnet18

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 2) Get the network!!
net = resnet18(pretrained=False).to(device)

# 3) Get data loaders!!
batch_size = 64
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 4) get optimizer
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

# 5) Define the objective function
misfit = torch.nn.CrossEntropyLoss().to(device)

#6) Start training!!!!
num_epochs = 10
for epoch in range(num_epochs):

    # Training Loop
    for i, (images, labels) in enumerate(trainloader):

        # Put data on gpu
        images = images.to(device)
        labels = labels.to(device)

        # Clear any old gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = net(images)
        loss = misfit(outputs, labels)

        # Backwards
        loss.backward()
        optimizer.step()

        # Compute stats

        if i%10==0:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).sum()/float(preds.shape[0])
            print('%6d,   Loss: %6.4e,   Acc: %6.4f' % (i, loss.item(), acc))

    # Validation Loop
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):

            # Put data on gpu
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = net(images)
            loss = misfit(outputs, labels)

            # Compute stats
            if i%10==0:
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).sum()/float(preds.shape[0])
                print('%6d,   Loss: %6.4e,   Acc: %6.4f' % (i, loss.item(), acc))