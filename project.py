import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader
import scipy.stats as stats
import copy
import numpy as np

# A class for our NN model.
class CIFAR_CNN(nn.Module):
    # Convolutional Neural Network for CIFAR-10 Classification
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            #https://stackoverflow.com/questions/69544256/how-padding-works-in-pytorch
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding = 1), # Out(32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding = 1), # Out (32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Out (16,16)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding = 1), # Out (16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Out (8,8)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding = 1), # Out (8,8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3), #randomly sets some activations to 0 at a 0.3 probability - to control overfitting 
            nn.Linear(2048, 10),
        )
        
        #https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        self.loss_func = nn.functional.cross_entropy
        self.optimizer = torch.optim.Adam(self.parameters())
        #schedule our learning rates to half if our loss and accuracy don't move and start to plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='min', factor=0.5, patience=2)

    def forward(self, x):
        return self.layers(x)
    
    def train_epoch(self, train_loader):
        """ Train on the entire dataset once (one epoch).
            Return value: (loss, accuracy)."""
        self.train()
        total_loss = 0
        correct = 0
        #https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
        hflipper = v2.RandomHorizontalFlip(p=0.5) 
        rcrop = v2.RandomCrop(32,4) #pad 4 pixels to each side of the image
        for data, target in train_loader:
            #the data has a 0.5 probability of randomly being chosen to be horizontally flipped
            data = hflipper(data)
            #will crop out a random 32x32 part of the image
            data = rcrop(data)
            # Evaluate the model and loss.
            output = self(data)
            loss = self.loss_func(output, target)
            # Gather some intermediate statistics
            total_loss += loss.item()*len(data)
            pred = output.argmax(dim=1)
            correct += (pred==target).sum()
            # Take a gradient descent step.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        N = len(train_loader.dataset)
        return total_loss/N, correct/N

    def assess(self, loader):
        """Evaluate the model on the given set.
           Return value: (loss, accuracy)."""
        self.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader: #will take train and val loader
                output = self(data)
                loss = self.loss_func(output, target)
                total_loss += loss.item()*len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        N = len(loader.dataset)
        return total_loss/N, correct/N


def train_model(model, min_epochs, max_epochs, patience=5):
    #calculate loss of train and validation sets
    train_loss, train_acc = model.assess(train_loader)
    val_loss, val_acc     = model.assess(val_loader)
    results = {'train_loss':[train_loss], 'train_acc':[train_acc],
               'val_loss':[val_loss], 'val_acc':[val_acc]}
    print(f"Prior to training, results={results}")
    epoch = 0
    epochs_since_improvement = 0
    best_val_acc = val_acc
    best_val_loss = val_loss
    training = True
    #epoch loop
    while training:
        epoch += 1
        print(f"*** Epoch {epoch}/{max_epochs} ***")
        train_loss, train_acc = model.train_epoch(train_loader)
        val_loss, val_acc     = model.assess(val_loader)
        model.scheduler.step(val_loss)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        if val_acc > best_val_acc or val_loss < best_val_loss:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            epochs_since_improvement = 0
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improvement += 1
        print(f"Train: loss={train_loss:0.4f}, acc={train_acc:0.4f}")
        print(f"  Val: loss={val_loss:0.4f}, acc={val_acc:0.4f}")
        print(f" best:      {best_val_loss:0.4f},     {best_val_acc:0.4f}, "+
              f"{epochs_since_improvement} epochs since improved")
        # Are we done training?
        if epoch >= min_epochs:
            if epoch>=max_epochs or epochs_since_improvement>=patience:
                training = False

    # Restore the state at the last improvement.
    print(f"Restoring model to state after epoch {best_epoch}.")
    model.load_state_dict(best_weights)
    model.eval()
    results['epoch_used'] = best_epoch
    return results

# main
# Make sure to edit the threads to your systems specifications
torch.set_num_threads(16) 
torch.set_num_interop_threads(8)
    
T = transforms.ToTensor()
fulltrain = datasets.CIFAR10('./data', train=True, transform=T, download=True)
# Partition the full training set into a (trainset,valset) pair.
train_size = int(len(fulltrain) * 0.8)
val_size   = len(fulltrain) - train_size
gen = torch.Generator().manual_seed(1) # For reproducability.
trainset, valset = random_split(fulltrain, [train_size, val_size], generator=gen)

#load the datasets
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader   = DataLoader(valset, batch_size=512)

#By turning train = True, it will load 50000 images but if false then it will load the 10k test set
testset = datasets.CIFAR10('./data', train=False, transform=T, download=True)
test_loader = DataLoader(testset, batch_size=512)

print("fullset", len(fulltrain)) #50000
print("train: ", len(trainset)) #40000
print("val: ", len(valset)) #10000
print("test: ", len(testset)) #10000

#http://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://discuss.pytorch.org/t/num-classes-in-dataset/158728/3
print(fulltrain.classes)
classes = fulltrain.classes


############################ initialize model ###################################
# Build and train the model.
model = CIFAR_CNN()
results = train_model(model, patience=8, min_epochs=30, max_epochs=50)
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

#https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""This will collect the predictions for each class under the testset and output it as a percentage"""
# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
test_loss, test_acc = model.assess(test_loader)
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")


################################################################
#calculate our confidence interval
# Roughly a 95% confidence interval estimate for true accuracy
# based on the validation results per epoch.
num_epochs = len(results["train_acc"])-1 # -1 b/c pre-train numbers @ pos. 0
z = stats.norm.ppf(0.95)
A = torch.tensor(results['val_acc'])
dev = z*(A*(1-A)/val_size)**(1/2)
fig,ax = plt.subplots(1,2,figsize=(10,5))
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# Plot the accuracy on train & validate sets by epoch:
x = [j for j in range(1, num_epochs+1)]
ax[0].plot(x, results["train_acc"][1:], color="black", label="train")
ax[0].plot(x, results["val_acc"][1:], color="blue", label="validation")
best = results['epoch_used']
y = results["val_acc"][best]
ax[0].plot(best, y, marker='o', color="green", label=f"{y:.4f}")
v0 = torch.tensor(results["val_acc"]) - dev
v1 = torch.tensor(results["val_acc"]) + dev
ax[0].fill_between(x, v0[1:], v1[1:], color="blue", alpha=0.1, label="95% CI estimate")
ax[0].axis([1,num_epochs,0.5,0.9])
ax[0].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
ax[0].legend(loc="lower right")
# Similarly with the loss:
ax[1].plot(x, results["train_loss"][1:], label="train")
ax[1].plot(x, results["val_loss"][1:], label="validation")
y = results["val_loss"][best]
ax[1].plot(best, y, marker='o', color="green", label=f"{y:.4f}")
ax[1].axis([1,num_epochs, 0, 1.3])
ax[1].set(title="Loss", xlabel="epoch", ylabel="loss")
ax[1].legend(loc="upper right")

#find the upper bound and lower bound percentages
#https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
import scipy.stats
confidence = 0.95  # Change to your desired confidence level
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
ci_length = z_value * np.sqrt((test_acc * (1 - test_acc)) / len(testset))
ci_lower = test_acc - ci_length
ci_upper = test_acc + ci_length
print(f"Confidence interval - lower: ({ci_lower:.4f}, upper: {ci_upper:.4f})")
plt.show()


#################################Visual######################################
#Let's see how our model is doing with the predictions alongside the label and image to compare our performance
samples = next(iter(val_loader))
with torch.no_grad():
    Y = model(samples[0]) #first batch of images predictions 
# Display the sample images, their predicted values, and actual values.
fig,ax = plt.subplots(8,8)
fig.subplots_adjust(hspace=0.5)
colors = {True:"green", False:"red"}
for k in range(64):
    X = samples[0][k].permute((1,2,0))
    y = int(samples[1][k])
    y_pred = int(Y[k].argmax())
    r = k//8
    c = k%8
    col = colors[y_pred == y]
    ax[r,c].imshow(X)
    ax[r,c].set_title(f"Pred: {classes[y_pred]}\nActual: {classes[y]}",color=col)
    ax[r,c].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.show()
    