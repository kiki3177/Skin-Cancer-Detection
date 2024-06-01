import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
import time
import random


import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# supress pytorch warnings
import warnings

warnings.filterwarnings("ignore")

timestamp = time.strftime("%m_%d_%H_%M_%S")

# set random seed
torch.manual_seed(916)
random.seed(916)
np.random.seed(916)

metadata_file = "data/HAM10000_metadata.csv"
image_folder = "data/HAM10000_images_merged"

mapping = {'akiec': 0,
           'bcc': 1,
           'bkl': 2,
           'df': 3,
           'mel': 4,
           'nv': 5,
           'vasc': 6
           }

train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

num_epochs = 10
batch_size = 6
learning_rate = 1e-4


# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available!")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available!")
else:
    device = torch.device("cpu")
    print("Use CPU only.")

original_label = pd.read_csv(metadata_file)
shuffled_label = original_label.sample(frac=1).reset_index(drop=True, inplace=False)

class SkinLesionDataset(Dataset):
    def __init__(self, shuffled_label, img_dataset_path, normalization=None, transform=None, target_transform=None):
        self.shuffled_label = shuffled_label
        self.lesion_id = self.shuffled_label.iloc[:, 0].tolist()
        self.image_id = self.shuffled_label.iloc[:, 1].tolist()
        self.dx = self.shuffled_label.iloc[:, 2].tolist()
        self.dx_type = self.shuffled_label.iloc[:, 3].tolist()
        self.age = self.shuffled_label.iloc[:, 4].tolist()
        self.sex = self.shuffled_label.iloc[:, 5].tolist()
        self.localization = self.shuffled_label.iloc[:, 6].tolist()

        self.label = []
        for type in self.dx:
            self.label.append(mapping[type])

        self.img_dataset_path = img_dataset_path
        self.img_paths = []
        for filename in self.image_id:
            self.img_paths.append(os.path.join(os.getcwd(), self.img_dataset_path, filename + ".jpg"))

        self.normalization = normalization
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        pixel = read_image(img_path)
        label = self.label[idx]
        if self.normalization:
            pixel = self.normalization(pixel)
        if self.transform:
            pixel = self.transform(pixel)
        if self.target_transform:
            label = self.target_transform(label)
        return pixel, label


def train_test_val_split(shuffled_label, type):
    total_samples = len(shuffled_label)
    train_samples = shuffled_label[0:int(train_ratio * total_samples)]
    test_samples = shuffled_label[int(train_ratio * total_samples): int((train_ratio + test_ratio) * total_samples)]
    val_samples = shuffled_label[int((train_ratio + test_ratio) * total_samples):
                                 int((train_ratio + test_ratio + val_ratio) * total_samples)]

    if type == 'train':
        train_dataset = SkinLesionDataset(shuffled_label=train_samples, img_dataset_path=image_folder,
                                          normalization=(lambda x: x / 255.0),
                                          transform=None,
                                          target_transform=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    if type == 'test':
        test_dataset = SkinLesionDataset(shuffled_label=test_samples, img_dataset_path=image_folder,
                                         normalization=(lambda x: x / 255.0),
                                         transform=None,
                                         target_transform=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_loader

    if type == 'val':
        val_dataset = SkinLesionDataset(shuffled_label=val_samples, img_dataset_path=image_folder,
                                        normalization=(lambda x: x / 255.0),
                                        transform=None,
                                        target_transform=None)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return val_loader


train_loader = train_test_val_split(shuffled_label, "train")
test_loader = train_test_val_split(shuffled_label, "test")
val_loader = train_test_val_split(shuffled_label, "val")


class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.resnet152 = models.resnet152(pretrained=True)
        in_features = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet152(x)


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


num_classes = len(mapping)
model = ResNet152(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batches = len(train_loader) * num_epochs

progress_bar = tqdm(total=total_batches)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


test_losses = []
test_accuracies = []
log_save_file_path = "training_logs.txt"
header = f"\nTime, Epoch, Train_Loss, Train_Accuracy, Val_Loss, Val_Accuracy\n"
method = "w"
with open(log_save_file_path, method) as f:
    f.write(header)


for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_predicted_labels = []
    train_true_labels = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        train_predicted_labels.extend(predicted_train.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())

        progress_bar.update(1)
        progress_bar.set_description(
            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}.")

    plot_confusion_matrix(train_true_labels, train_predicted_labels, classes=list(mapping.keys()))

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f"\nEpoch {epoch + 1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}%.")


    # Save model
    parent_dir = os.path.join("models", timestamp)
    os.makedirs(parent_dir, exist_ok=True)
    model_filename = os.path.join(parent_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_filename)


    # Validation
    model.eval()
    validation_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(inputs_val)
            validation_loss += criterion(outputs_val, labels_val).item()
            _, predicted_val = torch.max(outputs_val, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    validation_loss /= len(val_loader)
    validation_accuracy = 100 * correct_val / total_val
    print(f"Epoch {epoch + 1}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}%.")




# Test
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models", "05_14_12_55_23", "model_epoch_7.pth")))
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
test_predicted_labels = []
test_true_labels = []
with torch.no_grad():
    for inputs_test, labels_test in test_loader:
        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)
        outputs_test = model(inputs_test)
        test_loss += criterion(outputs_test, labels_test).item()
        _, predicted_test = torch.max(outputs_test, 1)
        total_test += labels_test.size(0)
        correct_test += (predicted_test == labels_test).sum().item()
        test_predicted_labels.extend(predicted_test.cpu().numpy())
        test_true_labels.extend(labels_test.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy}%.")

plot_confusion_matrix(test_true_labels, test_predicted_labels, classes=list(mapping.keys()))
