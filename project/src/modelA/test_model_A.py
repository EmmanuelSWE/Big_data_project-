from pathlib import Path 
import torch 
import torch.nn as nn 

from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from modelA.model_A import ModelA
from modelA.train_model_A import (
    split_dataset,
    WildlifeSpeciesDataset
)

import pandas as pd


def test_model():

    #laod device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # leoad checkpoint 
    checkpoint = (Path(__file__).resolve().parent/ "checkpoints" / "modelA_best.pth")

    if not checkpoint.is_file():
        raise FileNotFoundError(" pre trained model not found")

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

    # put in evaluate mode 
    model.eval()


    _,_, test_df = split_dataset()

    transform = ResNet50_Weights.DEFAULT.transforms()

    test_dataset = WildlifeSpeciesDataset(test_df,class_ids,transform)

    # load it the data into a loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    actual_labels = []
    predicted_labels = []

    # tesst loop
    with torch.inference_mode():
        for images, labels in  test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            actual_labels.extend(labels.tolist())
            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    # calc accuracy
    accuracy = accuracy_score(
        actual_labels,predicted_labels
    )

    print(f"Test images {len(actual_labels)}")
    print(f"Test accuracy {accuracy * 100 :.2f}%")


     
    # show precision and recall 
    report =  classification_report(
                actual_labels,predicted_labels,labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
    print(
        report 
    )

    # store confusion matrix
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names)))
    )

    print('\nConfusion matrix')
    print(f"class order: {class_names}")
    print(matrix)

    # create the result path
    results_path = Path(__file__).resolve().parent / "results"
    results_path.mkdir(parents = True, exist_ok = True)

    # save per species martices and macro.weigthed averages 
    report_rows = []


    for species in class_names + ["macro avg", "weighted avg"]:
        metric = report[species]

        report_rows.append({
            "species" : species,
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1-score": metric["f1-score"],
            "support": metric["support"]
        })

    pd.DataFrame(report_rows).to_csv(
            results_path/"modelA_test_metrics.csv",
            index=False
        )
    
        # store overall matrics seperatley 
    pd.DataFrame([{
            "checkpoint_epoch": checkpoint_model["epoch"],
            "test_images": len(actual_labels),
            "accuracy": accuracy
        }]
        ).to_csv(
            results_path/ "modelA_test_summary.csv",
            index=False
        )
    
        # save confusion marix 
    matrix_df = pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names
        )
    
    matrix_df.to_csv(
            results_path/"modelA_confusion_matix.csv",
            index_label="actual_species"
        )
    
        # save each test image's actual and predicted species 
    prediction_df = pd.DataFrame({
            "image_id": test_df["image_id"].tolist(),
            "actual_species": [
                class_names[label] for label in actual_labels
    
            ],
            "predicted_species":[
                class_names[label] for label in predicted_labels
            ]
        })
    
        #store the correct ones
    prediction_df["correct"]=(
            prediction_df["actual_species"] == prediction_df["predicted_species"]
        )
    
    prediction_df.to_csv(
            results_path/"modelA_test_predictions.csv",
            index=False
        )
    
    print("Testing complete")



if __name__ == "__main__":
    test_model()



