import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

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
                 patience=20,
                 checkpoint = None,
                 scheduler = None,
    ):
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
        self.start_epoch = 0
        self.train = {"loss":{}, "accuracy":{}}
        self.test = {"loss":{}, "accuracy":{}}
        self.scheduler = scheduler

        if checkpoint != None:
            self.early_stopping = checkpoint["early_stopping"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if (("scheduler_state_dict" in checkpoint.keys()) & (self.scheduler != None)):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.train = checkpoint["train_loss_acc"]
            self.test = checkpoint["test_loss_acc"]

    def accuracy(self,correct, total):
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
            loss_unweighted = torch.mean(loss_fn(y_pred,y))
            losses_train.append(loss_unweighted.detach().numpy())
            
            loss = torch.mean(w*loss_fn(y_pred,y))

            #Calculate accuracy
            correct += torch.sum(torch.round(y_pred) == y).detach().numpy()

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
            for x,y,w in test_dataloader:

                #Forwards pass
                y_pred = model(x)

                #Compute loss
                loss = torch.mean(loss_fn(y_pred,y))
                losses_test.append(loss.detach().numpy())

                #Calculate accuracy
                correct += torch.sum(torch.round(y_pred) == y).detach().numpy()

                #Add the batchsize to total
                total += x.shape[0]

        return (np.mean(losses_test), self.accuracy(correct,total))

    def __call__(self, path):
        """
        Run training of the model 
        
        :param path: Path to the directory, where the model/optimizer state and the loss/accuracies are saved
        :return:
        """
        
        #Initialize dictionaries for train/test losses and accuracies

        for i in range(self.start_epoch, self.num_epochs):
    
            #Get the test_loss and test_acc first so the losses and accuracy shows performance of the same model state
            test_loss, test_acc = self.test_step(self.test_dataloader, self.model, self.loss_fn)
            train_loss, train_acc = self.train_step(self.train_dataloader, self.optimizer, self.model, self.loss_fn)
            
            if self.scheduler != None:
                self.scheduler.step()

            self.test["loss"][i] = test_loss
            self.test["accuracy"][i] = test_acc
            self.train["loss"][i] = train_loss
            self.train["accuracy"][i] = train_acc
            
            #Early stopping mechanism
            if self.early_stopping["activate"]:
                if (self.early_stopping["ref_loss"] - test_loss) < self.early_stopping["min_delta"]:
                    self.early_stopping["count"] += 1
                else:
                    self.early_stopping["count"] = 0
                    self.early_stopping["ref_loss"] = test_loss            
            
            #Save state of optimizer + model to in case continue training
            if self.scheduler == None:
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": i,
                            "train_loss_acc": self.train,
                            "test_loss_acc": self.test,
                            "early_stopping": self.early_stopping,
                        }, path + "/" + "model_checkpoint.tar")

            else:
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict":self.scheduler.state_dict(),
                            "epoch": i,
                            "train_loss_acc": self.train,
                            "test_loss_acc": self.test,
                            "early_stopping": self.early_stopping,
                        }, path + "/" + "model_checkpoint.tar")

        
            print(f"Epoch: {i}, Train loss: {train_loss}, Test loss: {test_loss}")

            if self.early_stopping["count"] > self.early_stopping["patience"]:
                break
