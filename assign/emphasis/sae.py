# -----------------------------------------------------------------------------
# This script trains a supervised autoencoder model on the building fault
# detection dataset.
# The architecture of the model is as follows:
# 1. Encoder: 4 layers, each with ReLU activation
# 2. Decoder: 3 layers, each with ReLU activation
# 3. Classifier: 1 layer with softmax activation
# The model is trained using the Adam optimizer
#
# The architecture is based on the following paper:
# A data driven fault detection and diagnosis scheme for air handling units
# in building HVAC systems considering undefined states
# https://doi.org/10.1016/j.jobe.2020.102111
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# import train_test_split
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from datetime import datetime
import argparse


# Create dataset for training autoencoder
class BuildingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx, :-1].values, self.df.iloc[idx, -1]


# model definition
class SupervisedAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, nlabels, dropout, device="cpu"):
        super(SupervisedAutoEncoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), int(hidden_size / 3)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(hidden_size / 3), int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )
        self.classifer = nn.Sequential(
            nn.Linear(int(hidden_size / 3), nlabels),
            # nn.Softmax()
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        y = self.classifer(x)
        x = self.decoder(x)
        return x, y

    def batch_eval(self, batch, labels):
        x = batch
        y = labels
        x = x.to(self.device)
        y = y.to(self.device)
        x_hat, y_hat = self.forward(x)
        loss = ((x - x_hat) ** 2).mean()
        if torch.isnan(loss):
            print("nan encountered")
        class_loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss, class_loss


# implement early stopping mechanism to stop the training loop incase there is no significant
# change in validation loss
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # logic for early stopping
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def train_epoch(model, dataloader, optimizer):
    running_loss = 0.0
    running_class_loss = 0.0
    model.train()
    # for i, data in tqdm(enumerate(dataloader), desc="Training epoch"):
    for i, data in enumerate(dataloader):
        X = data[0].type(torch.float32)
        y = data[1].type(torch.LongTensor)
        loss, class_loss = model.batch_eval(X, y)
        total_loss = loss + class_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        running_class_loss += class_loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_class_loss = running_class_loss / len(dataloader)
    return epoch_loss, epoch_class_loss


def eval_metrics(y_true, y_pred):
    # using sklearn metrics
    # calculate precision, recall & f1 score
    # from sklearn.metrics import precision_recall_fscore_support

    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     y_true, y_pred, average="micro"
    # )
    from sklearn.metrics import (
        # confusion_matrix,
        # accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    return precision, recall, f1


def train_model(
    model, dataloader, valid_dataloader, optimizer, num_epochs=10, early_stopping=None
):
    train_losses = []
    val_losses = []
    # val_data = torch.tensor(df_normal_test.values, dtype=torch.float32)
    best_val_loss = float("inf")
    best_model = None
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_loss, epoch_class_loss = train_epoch(model, dataloader, optimizer)
        total_loss = epoch_loss + epoch_class_loss
        train_losses.append((epoch_loss, epoch_class_loss))
        model.eval()
        running_vloss = 0.0
        running_vclass_loss = 0.0
        with torch.no_grad():
            for i, val_data in enumerate(valid_dataloader):
                X = val_data[0].type(torch.float32)
                y = val_data[1].type(torch.LongTensor)
                val_loss, val_class_loss = model.batch_eval(X, y)
                if torch.isnan(val_loss):
                    print("nan encountered")
                running_vloss += val_loss.item()
                running_vclass_loss += val_class_loss.item()

        val_loss = running_vloss / len(valid_dataloader)
        val_class_loss = running_vclass_loss / len(valid_dataloader)
        tot_val_loss = val_loss + val_class_loss
        val_losses.append((val_loss, val_class_loss))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\
            Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f},\
                    Val Loss: {tot_val_loss:.4f}")
        if early_stopping is not None:
            if early_stopping(tot_val_loss):
                print("Early Stopping")
                break

        # keep track of best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_epoch = epoch

    return train_losses, val_losses, best_model, best_epoch, best_val_loss


def load_data():
    dataset = pd.read_csv("emphasis/building_fault_detection.csv")
    return dataset


def clean_data(dataset: pd.DataFrame):
    df_clean = dataset.copy()
    # fix space padding in column names
    columns = [c.strip() for c in df_clean.columns]
    df_clean.columns = columns
    # Convert all columns to numeric
    df_clean["AHU: Outdoor Air Temperature"] = pd.to_numeric(
        df_clean["AHU: Outdoor Air Temperature"], errors="coerce"
    )
    # Handle missing values. Drop rows with NaN
    df_clean = df_clean.dropna(axis="index")  # drop rows with NaN
    # Convert Datetime to datetime object
    df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"], format="%d-%m-%Y %H:%M")
    df_processed = df_clean.copy()

    df_processed["Labels"] = pd.NA
    # unfaulted normal days to 0
    df_processed.loc[df_processed["Datetime"].dt.day == 20, "Labels"] = 0
    df_processed.loc[df_processed["Datetime"].dt.day == 21, "Labels"] = 0
    df_processed.loc[df_processed["Datetime"].dt.day == 23, "Labels"] = 0
    df_processed.loc[df_processed["Datetime"].dt.day == 24, "Labels"] = 0
    # cooling coil fault 1
    df_processed.loc[df_processed["Datetime"].dt.day == 11, "Labels"] = 1
    df_processed.loc[df_processed["Datetime"].dt.day == 22, "Labels"] = 1
    # heating coil fault 2
    df_processed.loc[df_processed["Datetime"].dt.day == 12, "Labels"] = 2
    df_processed.loc[df_processed["Datetime"].dt.day == 14, "Labels"] = 2
    df_processed.loc[df_processed["Datetime"].dt.day == 15, "Labels"] = 2
    # oa damper fault 3
    df_processed.loc[df_processed["Datetime"].dt.day == 18, "Labels"] = 3
    df_processed.loc[df_processed["Datetime"].dt.day == 19, "Labels"] = 3

    df_processed["Labels"] = df_processed["Labels"].astype("category")

    # drop the datetime column and Supply fan status column
    df_processed = df_processed.drop(columns=["Datetime", "AHU: Supply Air Fan Status"])

    return df_processed


def split_data(df_processed: pd.DataFrame):
    # create training set with labels 0, 1, and 2
    df_train = df_processed[df_processed["Labels"].isin([0, 1, 2])]
    df_test = df_processed[df_processed["Labels"].isin([3])]

    # create train and valid splits, equally distributing the labels across the splits
    df_train_0 = df_train[df_train["Labels"] == 0]
    df_train_1 = df_train[df_train["Labels"] == 1]
    df_train_2 = df_train[df_train["Labels"] == 2]

    df_train_0, df_valid_0 = train_test_split(
        df_train_0, test_size=0.2, random_state=42
    )
    df_train_1, df_valid_1 = train_test_split(
        df_train_1, test_size=0.2, random_state=42
    )
    df_train_2, df_valid_2 = train_test_split(
        df_train_2, test_size=0.2, random_state=42
    )

    df_train = pd.concat([df_train_0, df_train_1, df_train_2])

    df_valid = pd.concat([df_valid_0, df_valid_1, df_valid_2])

    return df_train, df_valid, df_test


def create_dataset(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, batch_size=32
):
    # create datasets
    train_dataset = BuildingDataset(df_train)
    valid_dataset = BuildingDataset(df_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # create test dataset
    test_dataset = BuildingDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def create_model(input_size: int, hidden_size: int, nlabels: int, dropout: float):
    # automate device selection
    device = select_device()
    supervised_ae = SupervisedAutoEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        nlabels=nlabels,
        dropout=dropout,
        device=device,
    )
    supervised_ae.to(device)
    optimizer = optim.Adam(
        supervised_ae.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01
    )
    # optimizer = optim.SGD(supervised_ae.parameters(), lr=0.01, momentum=0.9)
    print(supervised_ae)

    return supervised_ae, optimizer, device


def select_device():
    # automate device selection
    device = None
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def plot_loss(train_losses, valid_losses):
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(valid_losses)), valid_losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.savefig("loss_plot.png")
    # plt.show()


def predict(model, test_loader):
    y_true = []
    y_pred = []
    x_true = []
    x_pred = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            X = test_data[0].type(torch.float32)
            y = test_data[1].type(torch.LongTensor)
            x_hat, y_hat = model(X)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.argmax(dim=1).cpu().numpy())
            x_true.append(X.cpu().numpy())
            x_pred.append(x_hat.cpu().numpy())
    return x_true, x_pred, y_true, y_pred


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    p = argparse.ArgumentParser()
    # train model ?
    p.add_argument("--train_model", type=str2bool, default=True)
    # model params
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--input_size", type=int, default=330)
    p.add_argument("--hidden_size", type=int, default=330)
    p.add_argument("--nlabels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=500)
    # early stopping params
    p.add_argument("--early_stopping", type=bool, default=True)
    p.add_argument("--patience", type=int, default=50)
    # optimizer params
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_model", type=bool, default=True)
    p.add_argument("--load_model", type=bool, default=False)
    p.add_argument("--model_path", type=str, default="sae_model.pth")

    args = p.parse_args()
    hp = vars(args)
    # load data
    ds = load_data()
    # clean data
    df = clean_data(ds)
    # split data
    df_train, df_valid, df_test = split_data(df)
    # create pytorchdataset
    train_loader, valid_loader, test_loader = create_dataset(
        df_train, df_valid, df_test
    )
    # create model
    supervised_ae, optimizer, device = create_model(
        input_size=df_train.shape[1] - 1,
        hidden_size=hp["hidden_size"],
        nlabels=3,
        dropout=hp["dropout"],
    )
    if hp["train_model"]:
        # train model
        train_losses, valid_losses, best_model, best_epoch, best_val_loss = train_model(
            supervised_ae,
            train_loader,
            valid_loader,
            optimizer,
            num_epochs=500,
            early_stopping=EarlyStopping(patience=50, delta=0.001),
        )
        print(f"Best epoch: {best_epoch}, Best Validation Loss: {best_val_loss}")

        # plot losses
        plot_loss(train_losses, valid_losses)

        # save best model
        print("Saving model to sae_model.pth")
        torch.save(best_model, "sae_model.pth")

    # load best model
    print("Loading model from sae_model.pth")
    supervised_ae.load_state_dict(torch.load("sae_model.pth"))

    # evaluate model
    x_true, x_pred, y_true, y_pred = predict(supervised_ae, valid_loader)

    # confusion matrix
    from sklearn.metrics import confusion_matrix

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    # calculate metrics
    precision, recall, f1 = eval_metrics(y_true, y_pred)
    print(f"Accuracy: {np.mean(np.array(y_true) == np.array(y_pred))}")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


if __name__ == "__main__":
    main()
