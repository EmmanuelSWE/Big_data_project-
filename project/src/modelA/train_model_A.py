
from pathlib import Path
import pandas as pd 
from contextlib import closing
import sqlite3
from animal_database.wildlife_config import DATABASE_PATH, SUB_DATASETS
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn as nn
from modelA.model_A import ModelA 
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
#fucntion to train model 
def load_training_records(db_path):
    # create and validate path
    ds_path = Path(db_path)

    if not ds_path.is_file():
        raise FileNotFoundError("train data base path not found")
    # connecct to db 

    with closing(sqlite3.connect(ds_path)) as database:
         # read data into data frame 
        
        query = 'select image_id,path,species,split from Animal_All_Species'
        data = pd.read_sql_query(query,database)

        identities =[]

        for species in SUB_DATASETS:
            tableName = f"species_{species}"

            query = f"select image_id, identity from {tableName}"

            species_data = pd.read_sql_query(query,database)

            identities.append(species_data)

        full_idnetities = pd.concat(identities, ignore_index= True)

        data = data.merge(
            full_idnetities,
            on = "image_id",
            how="left",
            validate="one_to_one"
        )

        if data["identity"].isna().any():
            raise ValueError("some images have no identity")
        


    #check the reccords
    print(data['species'].value_counts())

    return data


def split_dataset():
    dataset = load_training_records(DATABASE_PATH)

    valid_splits = {"train","val","test"}

    if dataset["split"].isna().any():
        raise ValueError("some images have no split assigned")

    if not set(dataset["split"]).issubset(valid_splits):
        raise ValueError("invalid split")

    train_df = dataset[
        dataset["split"] == "train"
    ].reset_index(drop=True)

    test_df = dataset[
        dataset["split"] == "test"
    ].reset_index(drop=True)

    val_df = dataset[
        dataset["split"] == "val"
    ].reset_index(drop=True)

    if any(len(part)== 0 for part in 
           [train_df, val_df,test_df]):
        raise ValueError("One or more dataset spilts")


    print("\nGlobal dataset spilt:")
    print(f"Train images : {len(train_df)}")
    print(f"test images : {len(test_df)}")
    print(f"val images : {len(val_df)}") 
    


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
        img = Image.open(row_data["path"])

        # convert to rgb 
        img = img.convert("RGB")

        # transform the image 
        img = self.transform(img)
        species = row_data["species"]
        label = self.Ids[species]

        return img, label


def calcualte_metrics(actual_labels, predticted_labels):
    accuracy = accuracy_score(
        actual_labels,predticted_labels
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels,
        predticted_labels,
        labels=[0,1,2,3],
        average="macro",
        zero_division=0
    )

    return{
        "accuracy" : accuracy,
        "precision": precision,
        "recall" : recall,
        "f1" : f1
    }


def train_one_epoch(model, train_loader, optimizer,loss_fn, device):
    # set to training mode
    
    model.train()
    total_loss = 0.0 
    correct = 0 
    total = 0 

    actual_lables = []
    predicted_lables = []

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

        actual_lables.extend(
            lable.detach().cpu().tolist()
        )

        predicted_lables.extend(
            predicted.detach().cpu().tolist()
        )

        
        total += lable.size(0)



    avg = total_loss/total
    metrics = calcualte_metrics(
        actual_lables,predicted_lables
    )
    return avg, metrics


def validate_model(model, val_loader, loss_fn, device):
    # set to training mode
    
    model.eval()
    total_loss = 0.0 
    correct = 0 
    total = 0 

    actual_lables = []
    predicted_lables = []


    with torch.no_grad():
        for images, lables in val_loader:
                
                img = images.to(device)
                lable = lables.to(device)
        
                # prediction 
                pred = model(img)
                loss = loss_fn( pred,lable)
        
                
        
                total_loss += loss.item() * lable.size(0) 
                predicted = pred.argmax(dim =1)
                actual_lables.extend(
                    lable.detach().cpu().tolist()
                )
                
                predicted_lables.extend(
                    predicted.detach().cpu().tolist()
                )
                total += lable.size(0)



    avg = total_loss/total
    metrics = calcualte_metrics(
        actual_lables,predicted_lables
    )
    return avg, metrics


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

    # recording the hsitory 
    history = []

    results_path = Path(__file__).resolve().parent / "results"
    results_path.mkdir(parents=True, exist_ok= True)



    for epoch in range(epochs):
        print(f"\nStarting epoc{epoch +1}/{epochs}")

        # train the model 
        loss, metrics = train_one_epoch(model,train_loader,optimizer,loss_fn,device)

        val_loss, val_metrics = validate_model(model,val_loader,loss_fn,device)

        history.append({
            "epoch" : epoch + 1,
            "train_loss" : loss,
            "train_accuracy" : metrics["accuracy"],
            "train_precision" : metrics["precision"],
            "train_recall" : metrics["recall"],
            "train_f1" : metrics["f1"],
            "val_loss" : val_loss,
            "val_accuracy" : val_metrics["accuracy"],
            "val_precision" : val_metrics["precision"],
            "val_recall" : val_metrics["recall"],
            "val_f1" : val_metrics["f1"],
        })

        print(f"Epoch {epoch + 1}")
        print(f"loss is {loss}")
        print(f"Training accuracy : {metrics['accuracy'] * 100:.2f}")

        print(f"Val_loss is {val_loss}")
        print(f"Val_Training accuracy : {val_metrics['accuracy'] * 100:.2f}")

        if val_loss < best_val_losss:
            best_val_losss = val_loss
            torch.save({
                "head_state_dict": model.baseModel.fc.state_dict(),
                "class_to_ids": class_to_ids,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"]
            },cpath
            )
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        results_path/ "modelA_training_history.csv",
        index=False
    )
        