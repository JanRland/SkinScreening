"""
This script contains the training pipeline for the skin cancer model. 

@author: Dmitry Degtyar
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
# check if running notebook and use notebook backend for tqdm progress bar
from tqdm import tqdm
import math


# ------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------YOUR MODEL START---------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet
        
        Note:
            Set right resnet size.
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19
        
        Note:
            Set right vgg size.
        """

        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        
        Note:
            Set right densenet size.
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# ------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------YOUR MODEL END-----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------DATALOADER START---------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
class HAMDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, transforms=None, selective=True, OR=5, input_size=224):
        """
        Args:
            csv_file (string): Pandas df with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = dataset

        self.OR = OR  # the label of the over represented class
        self.tf = transforms  # the input list of transforms
        self.selective = selective  # flag to apply transform to under rep classes only
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def to_categorical(y, num_classes=None, dtype="float32"):
        y = np.array(y, dtype="int")
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pil_image = Image.fromarray(self.df.iloc[idx].loc['image'])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(self.input_size),
                                        transforms.Normalize(
                                            mean=[0.764, 0.538, 0.562],
                                            std=[0.138, 0.159, 0.177],
                                        )
                                        ])
        img_tensor = transform(pil_image)

        # print(self.df.iloc[idx].loc['label'])
        category = self.df.iloc[idx].loc['label']
        disease = torch.tensor(category, dtype=torch.long)

        # Applying transforms
        if self.tf != None:
            if self.selective == True:  # Can choose to NOT apply augmentation on over rep classes
                if torch.argmax(disease) != self.OR:
                    img_tensor = self.tf(img_tensor)
            else:  # Or just apply aug to ALL classes and samples
                img_tensor = self.tf(img_tensor)

        return {'image': img_tensor.double(), 'disease': disease}


# ------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------DATALOADER END-----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------


# --------------SETTING DEVICE AND INIT NET WITH PARAMS--------------------------

batch_size = 128  # 512 is okay for resnet32 and resnet50, bs 256 is okay for densenet121 and densenet201
n_epochs = 50
num_gpus = 4

lrs = []
lr_low = 0.001
lr_up = 0.1
lr_start = 0.01
warmup_boarder = round(n_epochs * 0.1) - 1
warmup_step = (lr_up - lr_start) / warmup_boarder


# sigmoid learning annealing formula
def sigmoid_lr(epoch):
    new_lr = lr_low + (lr_up - lr_low) / (1 + math.exp(0.2 * (2 * (epoch - warmup_boarder + 1)) - 1))
    return new_lr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
net, input_size = initialize_model(model_name="resnet",
                                   num_classes=7,
                                   feature_extract=False,
                                   use_pretrained=True)
print(net)
net.to(device)
net = torch.nn.parallel.DataParallel(net, device_ids=list(range(num_gpus)), dim=0)

# def init_weights(m):
#    if isinstance(m, nn.Linear):
#        torch.nn.init.xavier_uniform_(m.weight)
#       m.bias.data.fill_(0.01)
# net.apply(init_weights)


# Image augmentation only for training
tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=15,
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.0),
                            shear=(10))
])

print(net.parameters())

optimizer = optim.SGD(net.parameters(), lr=lr_start)
# Reduce lr to 10 percent if test accuracy doesnt improve by 1%
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001,
#                                                      threshold_mode='rel',
#                                                       cooldown=0, min_lr=0, eps=1e-08, verbose=True)

loss_function = nn.CrossEntropyLoss()
loss_function.to(device)
net.double()

# -----------------------------END INIT------------------------------------------


# ----------------------LOAD DATASETS--------------------------------------------
df = pd.read_pickle(
    '/scratch_shared/degtyard/HAMdataframe.pickle')  # here comes the right path
# df = df.head(100)

print("dataframe loaded", flush=True)

# Create training & testing tensors for features & labels
X_train, X_val, y_train, y_val = train_test_split(df.image, df.label, test_size=0.2, random_state=42, stratify=df.label)

print(y_train.isnull().values.any())

# Create over sampler, and over sample ONLY the TRAIN dataframe
oversample = ROS(random_state=42)
X = np.array(X_train.values.tolist()).squeeze()

# reshape train dataset fro oversamler
reshaped_X = X.reshape(X.shape[0], -1)

# oversampling
oversampled_X, oversampled_y = oversample.fit_resample(reshaped_X, y_train)

# reshaping X back to the first dims
new_X = oversampled_X.reshape(-1, 450, 450, 3)
# print(new_X.shape)

# creating series from 4d numpy array
X_train = pd.Series([new_X[x, :, :, :] for x in range(new_X.shape[0])], dtype=object, name='image')
# new oversampled labels as train labels
y_train = pd.Series(oversampled_y)

y_val = pd.Series(y_val)

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
#  val_df.to_pickle("/scratch_shared/degtyard/alexnet_training/validation_data.pickle")
#  train_df.to_pickle("/scratch_shared/degtyard/alexnet_training/training_data.pickle")

# print(train_df.head(1))
# print(val_df.head(1))

train_dataset = HAMDataLoader(
    dataset=train_df,
    transforms=tf,
    input_size=input_size
)
print("Number of training samples:", len(train_dataset))

validation_dataset = HAMDataLoader(
    dataset=val_df,
    input_size=input_size
)
print("Number of training samples:", len(validation_dataset))

train_loader = DataLoader(
    train_dataset,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)

val_loader = DataLoader(
    validation_dataset,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)

# -------------------------LOAD DATASETS-----------------------------------------


# -------------------------START TRAINING----------------------------------------


torch.autograd.set_detect_anomaly(True)

acc_best = 0.
epoch_best = 0
train_acc = []
train_losses = []

val_acc = []
val_losses = []

###################
# train the model #
###################
print("Start of the training:")

for epoch in range(n_epochs):
    loader = tqdm(train_loader)
    losses = []  # logs avg loss per epoch
    accs = []  # logs avg acc per epoch
    correct = 0  # counts how many correct predictions
    count = 0  # counts how many samples

    if epoch < warmup_boarder:
        current_lr = lr_start + (warmup_step * epoch)  # track lr after warm up
        print(f"warmup learning rate = {current_lr}")

    else:
        if epoch == warmup_boarder:
            print("Warmup end")
        current_lr = sigmoid_lr(epoch)
        print(f"learning rate = {current_lr}")

    lrs.append(current_lr)
    optimizer.param_groups[0]['lr'] = current_lr

    net.train()
    for bidx, (data) in enumerate(loader):
        images, labels = data["image"].to(device), data["disease"].to(device)

        score = net(images)
        loss = loss_function(score, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(score, -1).detach()  # need to detach else bug
            correct += (pred == labels).sum().item()  # count how many correct
            count += len(labels)
            acc = correct / count  # accumated accuracy

            losses.append(loss.item())
            accs.append(acc)
            loader.set_description(f'TRAIN | epoch {epoch + 1}/{n_epochs} | acc {acc:.4f} | loss {loss.item():.4f}')

    train_acc.append(acc)
    train_losses.append(torch.tensor(losses).cpu().mean().item())

    ######################
    # validate the model #
    ######################

    net.eval()
    with torch.no_grad():

        loader = tqdm(val_loader)
        losses = []  # logs loss per minibatch
        accs = []  # logs running acc throughout one epoch

        correct = 0  # counts how many correct predictions in one epoch
        count = 0  # counts how many samples seen in one epoch

        for bidx, (data) in enumerate(loader):
            images, labels = data["image"].to(device), data["disease"].to(device)

            score = net(images)
            loss = loss_function(score, labels)

            pred = torch.argmax(score, -1).detach()  # need to detach else bug
            correct += (pred == labels).sum().item()  # count how many correct
            count += len(labels)

            acc = correct / count  # accumated accuracy

            loader.set_description(f'TEST | epoch {epoch + 1}/{n_epochs} | acc {acc:.4f} | loss {loss.item():.4f}')

            losses.append(loss.item())
            accs.append(acc)

    #  check whether new accuracy is the best
    if acc > acc_best:
        acc_best = acc
        epoch_best = epoch

    if (epoch % 5) == 0:
        #  save net parameter
        torch.save(net.state_dict(),
                   f"/scratch_shared/degtyard/resnet50_training/network_parameters_resnet50_ep_{epoch}.pt")

    val_acc.append(acc)
    val_losses.append(torch.tensor(losses).cpu().mean().item())

    #scheduler.step(torch.tensor(acc))  # reduce lr if test acc does not improve

    if optimizer.param_groups[0]['lr'] == lr_low:
        print(f'EARLY STOPPING! LR achieved lower bound  {lr_low}')
        break

loss_dataframe = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses, "train_acc": train_acc,
                               "val_acc": val_acc}, columns=["train_loss", "val_loss", "train_acc", "val_acc"])

loss_dataframe.to_pickle(
    '/scratch_shared/degtyard/resnet50_training/resnet50_losses.pickle')  # set the right path for saving losses

print(f"best achieved accuracy is {acc_best} in epoch {epoch}")

# -------------------------END TRAINING------------------------------------------
