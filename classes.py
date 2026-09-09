import torch



class Population():
    
#Variables
    device : torch.device
    
    #Start Input
    genSize: int
    inputNodes: int
    
    #Layers
    popLayers : list
    popBiasLayers : list
    activationLayer: list
    
    #Training Parameters
    trainType: str      #bp(backprop), mut(mutation)
    

    def __init__(self,genSize: int,inputNodes: int)-> None:
        #Variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Start Input
        self.inputNodes = inputNodes
        self.genSize = genSize
        
        #Layers
        self.popLayers = []
        self.popBiasLayers = []
        self.activationLayers = []
        
    
        
        
        
    def addLayer(self, newNodes: int, activation: str = None) -> None:
        if not self.popLayers: #check size of previous layer
            prevLayerN = self.inputNodes
        else:
            prevLayerN = self.popLayers[-1].shape[2]
            
        
        
        newLayer = torch.randn(self.genSize,prevLayerN,newNodes)
        newBiasLayer = torch.zeros(self.genSize,1,newNodes)

        self.popLayers.append(newLayer)
        self.popBiasLayers.append(newBiasLayer)
        self.activationLayers.append(activation)
        
    
    def setTrainParams(self): 
        '''sets all the parameters used in training the neural network'''
        
    
    def train(self,
              XT: torch.tensor,
              YT: torch.tensor,
              generations: int):
        
        #Forward pass
        
        outputpreLayer=[]           
        outputLayers=[input.unsqueeze(0)] #Format networkN x inputN x data
        
        for layer, bias in zip(self.popLayers,self.popBiasLayers):
            outputpreLayer.append(torch.matmul(outputLayers[-1],layer)+bias)
            #activation
            outputLayers.append(torch.relu(outputpreLayer[-1]))
        
        #----------------------------------
        #--------------TODO----------------
        #----------Calculate-Loss----------
        #----------------------------------
        
        #Calculate Loss
        match self.lossType:
            case "MSE": #Mean Squared Error
                Loss = (outputLayers[-1]-YT)**2 
                
                
            case _: #no losstype found
                raise ValueError("losstype not defined")

        
            
        #Backward pass
            #Neurons
        neuronGrad = [Loss] #bias gradient      #Format networkN x inputN x data
        
        for i in list(reversed(range(len(self.popLayers))))[:-1]:
            mask = outputpreLayer[i-1] >= 0    #Relu activation mask
            PostNeuron = torch.matmul(neuronGrad[0],torch.transpose(self.popLayers[i],1,2))
            neuronGrad.insert(0,PostNeuron*mask)
            
        #Backward pass
            #Weights

        weightGrad=[]   #weight gradient        #Format networkN x inputN x data x data
        for i in reversed(range(len(neuronGrad))):
            weightGrad.insert(0,torch.matmul(outputLayers[i].unsqueeze(3),neuronGrad[i].unsqueeze(2)))
          
            
        #----------------------------------
        #--------------TODO----------------
        #---Transform-Grads-To-Mutation----
        #-------------Tensor---------------

            
        
        