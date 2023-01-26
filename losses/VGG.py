import torchvision.models as models
import torch.nn as nn
import torch.cuda as cuda

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','1','2','3'] 
        self.layer_weights= [0.0025,0.125,0.0025,0.125]
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:4] #model will contain the first5git layers
        if cuda.is_available():
            self.model = self.model.cuda()
    
   
    #x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        #initialize an array that wil hold the activations from the chosen layers
        features=[]
        style_features=[]
        #Iterate over all the layers of the mode
        i = 0
        for layer_num, layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)
                style_features.append(x*self.layer_weights[i])
                i +=1
                
        return features, style_features
    