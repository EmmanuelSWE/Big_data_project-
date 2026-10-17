from pathlib import Path 
import torch 
import torch.nn as nn 
from tkinter import Tk, filedialog

from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from PIL import Image
from modelA.model_A import ModelA
from modelA.train_model_A import (
    split_dataset,
    WildlifeSpeciesDataset
)

def load_model(checkpoint,device):
    checkpoint_model = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )
    
    class_ids = checkpoint_model["class_to_ids"]
    
    class_names = sorted(
        class_ids,
        key = class_ids.get
    )
    
    print(f"testing checkpoint form epoch: {checkpoint_model['epoch']}" )
    model = ModelA().to(device)
    
    #restore classification head 
    model.baseModel.fc.load_state_dict(
        checkpoint_model["head_state_dict"]
    )
    
    # put in evaluate mode du
    model.eval()

    return model, class_names


def predict_species(model,class_names,image,device):
    transform = ResNet50_Weights.DEFAULT.transforms()

    img = image.convert("RGB")
        

    img = transform(img).unsqueeze(0)

    img = img.to(device)
    with torch.inference_mode():
        pred = model(img)
        probablity = torch.softmax(pred,dim=1)

        prediction = torch.argmax(probablity, dim=1).item()
        confindence =  probablity[0,prediction].item()

    label = class_names[prediction]
    

    return label, confindence

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = (Path(__file__).resolve().parent/"checkpoints"/"modelA_best.pth")
    model,class_names = load_model(checkpoint,device)

    root = Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="select an image",
        filetypes=[("Images","*.jpg *.png *.jpeg")]
    )

    root.destroy()

    if not image_path:
        raise ValueError

    with Image.open(image_path) as image:
        lable, confidence = predict_species(model,class_names,image,device)

    print(f"Predicted : {lable} whith confidence {confidence * 100:.2f}%")