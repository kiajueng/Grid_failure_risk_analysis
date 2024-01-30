import torch
import torch.nn as nn

class MLP_binary(nn.Module):

    def __init__(self, input_size, hidden_sizes, hidden_act_fns, output_act_fn, output_size=2, dropout=True, dropout_prob=0.2):
        """Initialize the DNN model architecture
        
        :param input_size: Number of input features
        :param n_sizes: List with number of nodes in each hidden layer
        :param hidden_act_fns: List with activation functions applied on each layer
        :param output_act_fn: Output activation function
        :param output_size: Number of outputs
        :param dropout: Boolean to choose if dropout between layer should be used or not 
        :param dropout_prob: Dropout probability is the probability for a unit to drop out
        :return: 
        """
        #Initialize internal module state
        super(MLP_binary, self).__init__()

        #Initialize list, where the layers are appended to
        layers = []

        #Put input_size as first element and output size as last element into hidden sizes
        hidden_sizes.insert(0, input_size)
        hidden_sizes.append(output_size)

        #Append output_act_fn to the hidden_act_fns
        hidden_act_fns.append(output_act_fn)
        
        #Build each layer and append to layers
        for layer in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[layer],hidden_sizes[layer+1]))
            layers.append(hidden_act_fns[layer])

            #Check if dropout should be used and if yes add the dropout layer
            if dropout and (layer < len(hidden_sizes) - 2):
                layers.append(nn.Dropout(dropout_prob))

        #Cast list to nn.Sequential so each time calling model, every thing in layers is called sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Define forward method for MLP_binary
        
        :param x: Input data to the model
        :return self.model(x): Returns the data piped through the model
        """
        
        return self.model(x)
