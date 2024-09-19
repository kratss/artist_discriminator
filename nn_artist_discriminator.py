import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import os
import pandas as pd
from torchvision.io import read_image
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = "./data/comics/train"
        self.transform = transform
        self.images = []
        self.labels = []
    
        # Assuming you have a folder structure like:
        # root_dir/
        # ├── class_1/
        # │   ├── img1.jpg
        # │   └── img2.jpg
        # └── class_2/
        #     ├── img1.jpg
        #     └── img2.jpg
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

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to the desired size
    transforms.ToTensor(),
])

# Create an instance of your dataset
my_data = CustomDataset(root_dir='/datasets/d/ml/data/comics/train', transform=transform)
my_test_data = CustomDataset(root_dir='/datasets/d/ml/data/comics/test', transform=transform)




# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("CUDA is not available. Using CPU.")
#device = torch.device("cpu")
sample_input = torch.randn(1, 3, 28, 28).to(device)  # Create a sample input tensor
print("Sample input device:", sample_input.device)


batch_size = 100

# Create a DataLoader
my_dataloader = DataLoader(my_data, batch_size=batch_size, shuffle=True)
my_test_dataloader = DataLoader(my_test_data, batch_size=batch_size, shuffle=True)


print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
for inputs, labels in my_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
'''
# Check the size of dataloaders
print("my_dataloader: ")
for X, y in my_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    
print("my_test_dataloader: ") 
for X, y in my_test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
'''

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# Reduce LR every 5 epochs by a factor of 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, \
                                      step_size=7, gamma=0.5) 
running_train_accuracy = []
running_test_accuracy  = []

import torch.utils.bottleneck as bottleneck


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
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

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    running_test_accuracy.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(my_dataloader, model, loss_fn, optimizer)
    test(my_test_dataloader, model, loss_fn)
print("Done!")


import matplotlib.pyplot as plt
plt.plot(running_test_accuracy)


torch.save(model.state_dict(), "model_xkcd.pth")
print("Saved PyTorch Model State to model_xkcd.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model_xkcd.pth"))

# Define the classes
classes = [
    "xkcd",
    "pbf"
]

# Set the model to evaluation mode
model.eval()

# Get a single sample from the test dataset
x, y = my_test_data[16][0], my_test_data[16][1]

# Add a batch dimension and move to the appropriate device
x = x.unsqueeze(0).to("cpu")  # Shape becomes (1, 3, 28, 28)

# Make predictions
with torch.no_grad():
    pred = model(x)
    predicted = classes[pred[0].argmax(0).item()]  # Get the predicted class
    actual = classes[y]  # Use y directly since it's an int
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# Assuming 'model' is your trained model
model_weights = model.state_dict()

# Print the weights
for name, param in model_weights.items():
    print(f"Layer: {name}, Weights: {param.shape}")
    
    
    
# Assuming 'model' is your trained model
model_weights = model.state_dict()

# Print the entire state_dict (optional, can be large)
# print(model_weights)

# Example: Print weights and biases of the first linear layer
# Adjust the key names based on your model architecture
print("Weights of the first linear layer:")
print(model_weights['linear_relu_stack.0.weight'])  # Access weights
print("Biases of the first linear layer:")
print(model_weights['linear_relu_stack.0.bias'])    # Access biases

# Example: Print weights and biases of the second linear layer
print("Weights of the second linear layer:")
print(model_weights['linear_relu_stack.2.weight'])  # Access weights
print("Biases of the second linear layer:")
print(model_weights['linear_relu_stack.2.bias'])    # Access biases



