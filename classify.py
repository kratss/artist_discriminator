import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
         
        valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_file in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_file)
                    # Check if the file is an image
                    if os.path.isfile(img_path) and os.path.splitext(img_file)[1].lower() in valid_image_extensions:
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

device = 'cuda'
model = NeuralNetwork().to(device)
model = NeuralNetwork()
model.load_state_dict(torch.load("model_xkcd.pth"))
print(model)

model.eval() # set model to evaluation mode
my_data = CustomDataset(root_dir='./images', transform=transform)
x = my_data[0][0]
classes = ("pbf", "xkcd")

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for idx in range(len(my_data)):
    x,_ = my_data[idx]
    x = x.unsqueeze(0).to("cpu")             
    with torch.no_grad():
        pred = model(x)
        pred2 = torch.sigmoid(pred).item()  # Get the confidence score
        predicted = 1 if pred2 >= 0.5 else 0  # Use 0.5 as the threshold for binary classification
        imshow(torchvision.utils.make_grid(x))
        plt.title(f"Predicted: {classes[predicted]}, Confidence: {pred2*100:.2f}%")
        plt.axis('off')  # Hide axes
        plt.show()  # Display the image
