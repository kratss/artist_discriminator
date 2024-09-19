import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F


print("Checking GPU availability")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
sample_input = torch.randn(1, 3, 28, 28).to(device)
print("Sample input device:", sample_input.device)


print("\nLoading, resizing, and normalizing images")
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = "./data/comics/train"
        self.transform = transform
        self.images = []
        self.labels = []
         
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_file in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_file)
                    self.images.append(img_path)
                    self.labels.append(label)   
           
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if needed
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((56, 56)),  # Resize to the desired size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

my_data = CustomDataset(\
    root_dir='/datasets/d/data/comics/train',\
    transform=transform)
my_test_data = CustomDataset(\
    root_dir='/datasets/d/data/comics/test', \
    transform=transform)

batch_size = 32
my_dataloader = DataLoader(my_data, batch_size=batch_size, shuffle=True)
my_test_dataloader = DataLoader(my_test_data, batch_size=batch_size, shuffle=True)


print("\nLoading to device. If this device is not as expected, evaluate your installation")
for inputs, labels in my_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")


print("\nInitializing network model")
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) #outputs batchsize, 16, 56 56
        self.conv2 = nn.Conv2d(16, 32, 3) #outputs batchsize, 32 56 56
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #half img dim
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 25 * 25, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 1) #binary classifier   
        )

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

print("\nSetting loss function, optimizer, scheduler")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, \
                                      step_size=4, gamma=0.5) 
print("\nPrinting data sample")
classes = ("xkcd", "pbf")

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
my_iter = iter(my_dataloader)
images, labels = next(my_iter)

# show images
imshow(torchvision.utils.make_grid(images))


print("\nDefining train class")
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.unsqueeze(1).float()
        pred = model(X)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

print("\nDefining test class")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(1).float()
            pred = model(X)
            pred_prob = torch.sigmoid(pred)
            pred_binary = (pred_prob > 0.5).float()
            correct += (pred_binary == y).sum().item()
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    correct /= size
    running_test_accuracy.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("\nBeginning training")
running_train_accuracy = []
running_test_accuracy  = []
epochs = 7
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(my_dataloader, model, loss_fn, optimizer)
    test(my_test_dataloader, model, loss_fn)
plt.plot(running_test_accuracy)

print("\nSaving PyTorch Model State to artist_discriminator.pth")
torch.save(model.state_dict(), "artist_discriminator.pth")
