import timm
import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadTimmModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes1=6, num_classes2=3):
        super(MultiHeadTimmModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        feature_dim = self.backbone.num_features
        
        # Head1 (cleanliness status)
        self.head1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes1)
        )
        
        # Head2 (cover status)
        self.head2 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes2)
        )
    
    def forward(self, x):
        features = self.backbone(x)

        head1_output = self.head1(features)
        head2_output = self.head2(features)
        
        return head1_output, head2_output