import torch
import torch.nn as nn
import torchvision.models  as models

# const input shape and output shape
MODEL_INPUT = (3,224,224) # 224x224 RGB images
MODEL_OUTPUT = 4 # 4 species label
# making the first model using ResNet50
class ModelA(nn.Module):
    '''
        This class represents the ResNet50 model belonging to the ResNet family architecture.
        It is a pretrained model consisting of 50 layers and was trained on the ImageNet dataset.  
    '''
    def __init__(self):

        super().__init__()

        # baseline model
        self.baseModel = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # freeze model 
        for param in self.baseModel.parameters():
            param.requires_grad = False

        # get input features 
        features = self.baseModel.fc.in_features

        # replace output layer
        self.baseModel.fc = nn.Sequential(nn.Linear(features,MODEL_OUTPUT))


       

    def forward(self,x):
        x = self.baseModel(x)
        return x 
    

