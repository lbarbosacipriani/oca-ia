from torch import nn


class ClassifierSimpleMultiLabel(nn.Module):    
    def __init__(self, num_classes=5):
        super(ClassifierSimpleMultiLabel, self).__init__()
        # Where we define all the parts of the model
        #self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        enet_out_size = int(32*(250*250)/25)      # Make a classifier

        #self.base_model = timm.create_model('vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k',num_classes=32)
        self.model = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3)), ## 3, 256,256 -> 32, 254,254
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)), ## 32, 254,254 -> 64, 252,252 
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3)), ## 32, 250,250 ,
        nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes), nn.Sigmoid())
    
         # saida como linear 
    
    def forward(self, x):
        # Connect these parts and return the output
        #x = self.features(x)
        output = self.model(x)
        return output
    

class ClassifierSimpleSingleLabell(nn.Module):    
    def __init__(self, num_classes=5):
        super(ClassifierSimpleSingleLabell, self).__init__()
        # Where we define all the parts of the model
        #self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        enet_out_size = int(32*(250*250)/25)      # Make a classifier

        #self.base_model = timm.create_model('vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k',num_classes=32)
        self.model = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3)), ## 3, 256,256 -> 32, 254,254
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)), ## 32, 254,254 -> 64, 252,252 
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3)), ## 32, 250,250 ,
        nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes), nn.Sigmoid())
    
         # saida como linear 
    
    def forward(self, x):
        # Connect these parts and return the output
        #x = self.features(x)
        output = self.model(x)
        return output