import torch
import torch.nn as nn
import torchvision.models  as models

# const input shape and output shape
MODEL_INPUT = (3,224,224) # 224x224 RGB images
MODEL_OUTPUT = 4 # 4 species label

'''
This class defines Model A, which classifies wildlife images into
four species: dog, hyena, leopard and macaque.

It uses a ResNet50 model pretrained on ImageNet and replaces its
original classification layer with a layer for four species.
'''
class ModelA(nn.Module):
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


       
    '''
    This function defines the forward pass of Model A.

    The input image passes through the ResNet50 network, which
    extracts visual features and produces four classification scores.

    These scores are returned for training or prediction.
    '''
    def forward(self,x):
        x = self.baseModel(x)
        return x 
    

