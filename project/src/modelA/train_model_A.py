
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn as nn
from modelA.model_A import ModelA


# Same data source as Model B (project/src/model_b/dataset.py): a metadata.csv with
# a "split" column that already assigns train/test. Using the same file and the same
# split for both models means they're evaluated on the same leakage-safe split instead
# of Model A inventing its own random one.
METADATA_FILE = "data/metadata.csv"
ROOT_FOLDER = "data"
SPECIES_LIST = ["hyena", "leopard", "macaque", "dog"]  # must match model_b/train_all.py

# Column order of metadata.csv when it has no header row (same convention as
# model_b/dataset.py's read_metadata).
COLUMN_NAMES = ["identity", "path", "date", "orientation", "species", "split", "dataset", "extra"]


def read_metadata(metadata_file):
    table = pd.read_csv(metadata_file, dtype=str, keep_default_na=False)
    if "species" in table.columns and "split" in table.columns:
        return table  # the file has its own header row

    table = pd.read_csv(
        metadata_file, header=None, names=COLUMN_NAMES, usecols=range(len(COLUMN_NAMES)),
        dtype=str, keep_default_na=False,
    )
    return table


def load_training_records(metadata_file=METADATA_FILE, root_folder=ROOT_FOLDER, species_list=SPECIES_LIST):
    table = read_metadata(metadata_file)

    table = table[table["species"].isin(species_list)].copy()
    if len(table) == 0:
        raise ValueError(f"No rows found for species {species_list} in {metadata_file}")

    table["full_path"] = table["path"].apply(lambda p: os.path.join(root_folder, p))
    file_exists = table["full_path"].apply(os.path.exists)
    if (~file_exists).sum() > 0:
        print(f"skipping {(~file_exists).sum()} rows whose image file is missing")
    table = table[file_exists]

    # check the records
    print(table['species'].value_counts())

    return table


def split_dataset(val_fraction=0.15):
    """
    Uses metadata.csv's own train/test split (the same one Model B uses) instead of
    inventing a new one. A validation slice is carved out of the train split only, by
    individual, so no individual appears in both train and val.
    """
    table = load_training_records()

    train_full = table[table["split"] == "train"].copy()
    test_df = table[table["split"] == "test"].copy()

    train_list = []
    val_list = []

    for species in SPECIES_LIST:
        species_train = train_full[train_full["species"] == species]

        individuals = species_train["identity"].drop_duplicates().tolist()
        train_individuals, val_individuals = train_test_split(
            individuals, test_size=val_fraction, random_state=42
        )

        train_list.append(species_train[species_train["identity"].isin(train_individuals)])
        val_list.append(species_train[species_train["identity"].isin(val_individuals)])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)

    return train_df, val_df, test_df


class WildlifeSpeciesDataset(Dataset):
    def __init__(self,dataframe, class_of_ids,transform):
        self.dataframe = dataframe 
        self.Ids = class_of_ids 
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,index):
        row_data = self.dataframe.iloc[index]

        # open image
        img = Image.open(row_data["full_path"])

        # convert to rgb 
        img = img.convert("RGB")

        # transform the image 
        img = self.transform(img)
        species = row_data["species"]
        label = self.Ids[species]

        return img, label





def train_one_epoch(model, train_loader, optimizer,loss_fn, device):
    # set to training mode
    
    model.train()
    total_loss = 0.0 
    correct = 0 
    total = 0 

    for layer in model.baseModel.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


    for images, lables in train_loader:
        optimizer.zero_grad()
        img = images.to(device)
        lable = lables.to(device)

        # prediction 
        pred = model(img)
        loss = loss_fn( pred,lable)

        # bakcpropagations 
        loss.backward()
        
        optimizer.step()
        

        total_loss += loss.item() * lable.size(0) 
        predicted = pred.argmax(dim =1)
        correct += (predicted == lable).sum().item()
        total += lable.size(0)



    avg = total_loss/total
    accuracy = correct/total 
    return avg, accuracy


def validate_model(model, val_loader, loss_fn, device):
    # set to training mode
    
    model.eval()
    total_loss = 0.0 
    correct = 0 
    total = 0 

 


    with torch.no_grad():
        for images, lables in val_loader:
                
                img = images.to(device)
                lable = lables.to(device)
        
                # prediction 
                pred = model(img)
                loss = loss_fn( pred,lable)
        
                
        
                total_loss += loss.item() * lable.size(0) 
                predicted = pred.argmax(dim =1)
                correct += (predicted == lable).sum().item()
                total += lable.size(0)



    avg = total_loss/total
    accuracy = correct/total 
    return avg, accuracy


if __name__ == "__main__":

        
    ## creating the loaders 

    train_df, val_df, test_df = split_dataset()

    class_names = sorted(
        train_df["species"].drop_duplicates().tolist()
    )

    class_to_ids = {
        species: index
        for index, species in enumerate(class_names)
    }

    transform = ResNet50_Weights.DEFAULT.transforms()


    test_dataset = WildlifeSpeciesDataset(test_df,class_to_ids,transform)
    train_dataset = WildlifeSpeciesDataset(train_df,class_to_ids,transform)
    val_dataset = WildlifeSpeciesDataset(val_df,class_to_ids,transform)

    # the loaders 
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle= True,
        num_workers=0
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle= False,
        num_workers=0
    )



    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle= False,
        num_workers=0
    )




    ### setting up the training 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    print(f"Trainging device is {device}")

    #Create the model A 
    model = ModelA().to(device)


    loss_fn = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.baseModel.fc.parameters(),
        lr = 0.001,
        weight_decay= 0.0
    )

    checkpoint = Path(__file__).resolve().parent / "checkpoints"

    checkpoint.mkdir(parents=True, exist_ok=True)
    cpath = checkpoint/"modelA_best.pth"
    best_val_losss = float("inf")
    epochs = 5

    for epoch in range(epochs):
        print(f"\nStarting epoc{epoch +1}/{epochs}")

        # train the model 
        loss, accuracy = train_one_epoch(model,train_loader,optimizer,loss_fn,device)

        val_loss, val_accuracy = validate_model(model,val_loader,loss_fn,device)

        print(f"Epoch {epoch + 1}")
        print(f"loss is {loss}")
        print(f"Training accuracy : {accuracy * 100:.2f}")

        print(f"Val_loss is {val_loss}")
        print(f"Val_Training accuracy : {val_accuracy * 100:.2f}")

        if val_loss < best_val_losss:
            best_val_losss = val_loss
            torch.save({
                "head_state_dict": model.baseModel.fc.state_dict(),
                "class_to_ids": class_to_ids,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            },cpath
            )

        