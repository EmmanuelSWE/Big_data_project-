
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






#fucntion to train model 
def load_training_records(db_path):
    # create and validate path
    ds_path = Path(db_path)

    if not ds_path.is_file():
        raise FileNotFoundError("train data base path not found")
    # connecct to db 

    with closing(sqlite3.connect(ds_path)) as database:
         # read data into data frame 
        
        query = 'select image_id,path,species from Animal_All_Species'
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
    # create 3 empty lists 
    train_list = []
    val_list = []
    test_list = []
    # loop through each 
    
    dataset = load_training_records(DATABASE_PATH)

    for species in dataset["species"].drop_duplicates().tolist():
            # go throguh tits identites and split them 70 15 15
           specied_data =  dataset[dataset["species"]== species]

           individuals = specied_data["identity"].drop_duplicates().tolist()
           train, temp = train_test_split(individuals, test_size=0.30, random_state= 42)
           val,test = train_test_split(temp,test_size=0.50, random_state=42)

           train_data = specied_data[specied_data["identity"].isin(train)]
           test_data = specied_data[specied_data["identity"].isin(test)]
           val_data = specied_data[specied_data["identity"].isin(val)]

           train_list.append(train_data)
           test_list.append(test_data)
           val_list.append(val_data)
                         

        #combine the datasets per species 
    train_df = pd.concat(train_list,ignore_index= True)
    test_df = pd.concat(test_list,ignore_index= True)
    val_df = pd.concat(val_list,ignore_index= True)

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

        