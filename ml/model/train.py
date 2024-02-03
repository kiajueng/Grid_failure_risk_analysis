import torch
import torch.nn as nn
from tqdm import tqdm

class training():

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 optimizer,
                 model,
                 loss_fn,
                 num_epochs,
                 early_stopping=True,
                 min_delta=1e-2,
                 patience=30):
        """Initiaize parameters for running the training
        
        :param train_dataloader: Pytorch dataloader for the training data
        :param test_dataloader: Pytorch dataloader for the test data
        :param optimizer: optimizer, with which the parameters of the model are updated
        :param model: The DNN model
        :param loss_fn: The loss function, to measure the loss 
        :param num_epochs: Number of epochs to run during training
        :return:
        """
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.early_stopping = {"activate":early_stopping, "ref_loss":np.inf, "count":0, "min_delta": min_delta, "patience": patience}

    def accuracy(correct, total):
        """Function to calculate the accuracy
        
        :param correct: Number of correctly classified samples
        :param total: total number of samples
        :return correct/total: Accuracy value for all predicted samples
        """
        
        return correct/total 

    def train_step(self, train_dataloader, optimizer, model, loss_fn):
        """One training step 
        
        :param dataloader: Pytorch dataloader with the train data
        :param optimizer: optimizer used for updating the weights
        :param model: The DNN model
        :param loss_fn: Loss function
        :return (loss,accuracy): Returns a tuple with the mean loss and mean accuracy over all the batches 
        """
        
        model.train()
        
        losses_train = []
        correct = 0
        total = 0

        for x,y,w in train_dataloader:

            #Delte the gradients from last training iteration
            optimizer.zero_grad()

            #Forwards pass
            y_pred = model(x)

            #Compute loss
            loss_unweighted = loss_fn(y_pred,y)
            losses_train.append(loss_unweighted.detach().numpy())

            loss = torch.mean(w*loss_fn(y_pred,y,reduction=None))

            #Calculate accuracy
            correct += torch.sum(torch.argmax(y_pred,dim=1) == y).detach().numpy()

            #Add the batchsize to total
            total += x.shape[0]

            #Backward pass to calculate the gradients and update weights
            loss.backward()
            optimizer.step()

        return (np.mean(losses_train), self.accuracy(correct,total))

    def test_step(self,test_dataloader,model,loss_fn):
        """Compute the loss and accuracy on the validation set

        :param optimizer: optimizer used for updating the weights
        :param model: The DNN model
        :param loss_fn: Loss function
        :return (loss,accuracy): Returns a tuple with the mean loss and mean accuracy over all the batches         
        """
        
        model.eval()

        losses_test = []
        correct = 0
        total = 0

        with torch.no_grad():
            for x,y in test_dataloader:

                #Forwards pass
                y_pred = model(x)

                #Compute loss
                loss = loss_fn(y_pred,y)
                losses_test.append(loss.detach().numpy())

                #Calculate accuracy
                correct += torch.sum(torch.argmax(y_pred,dim=1) == y).detach().numpy()

                #Add the batchsize to total
                total += x.shape[0]

        return (np.mean(losses_test), accuracy(correct,total))

    def run_training(self, path):
        """
        Run training of the model 
        
        :param path: Path to the directory, where the model/optimizer state and the loss/accuracies are saved
        :return:
        """
        
        #Initialize dictionaries for train/test losses and accuracies
        train_losses = {}
        train_accs = {}

        test_losses = {}
        test_accs = {}

        for i in tqdm(range(self.num_epoch)):
            print("Epoch: ",i)
            
            #Get the test_loss and test_acc first so the losses and accuracy shows performance of the same model state
            test_loss, test_acc = self.test_step(self.test_dataloader, self.model, self.loss_fn0)
            train_loss, train_acc = self.train_step(self.train_dataloader, self.optimizer, self.model, self.loss_fn)
            
            test_losses[i] = test_loss
            test_accs[i] = test_acc
            train_losses[i] = train_loss
            train_accs[i] = train_acc
            
            #Early stopping mechanism
            if self.early_stopping["activate"]:
                if (self.early_stopping["ref_loss"] - test_loss) < self.early_stopping["min_delta"]:
                    self.early_stopping["count"] += 1

                    if self.early_stopping["count"] > self.early_stopping["patience"]:
                         break
                else:
                    self.early_stopping["count"] = 0
                    self.early_stopping["ref_loss"] = test_loss
                
        #Put loss and accuracy dictionaries in one dictionary, for test and train respectively
        train,test = {}, {}

        train["loss"] = train_losses
        train["accuracy"] = train_accs

        test["loss"] = test_losses
        test["accuracy"] = test_accs
        
        #Safe the loss/accuracy dictionaries
        with open(path + "/" + 'train_loss_acc.pkl', 'wb') as fp:
            pickle.dump(train, fp)
            print('Train dictionary saved successfully to file')

        with open(path + "/" + 'test_loss_acc.pkl', 'wb') as fp:
            pickle.dump(test, fp)
            print('Test dictionary saved successfully to file')

        #Save state of optimizer + model to in case continue training
        torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    }, path + "/" + "model_checkpoint.tar"
                   
        
