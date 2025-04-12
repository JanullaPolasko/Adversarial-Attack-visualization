#this file contain function for training using pytorch lighting 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from networks import  get_dataset_mapping
from datapath import my_path
from datasets import load_data
import time
import subprocess
from visualization_utils import compute_dev_plot
import numpy as np

class Model(pl.LightningModule):
    """
    PyTorch Lightning model class that encapsulates the training and validation logic,
    including model initialization, forward pass, loss calculation, optimizer configuration,
    and tracking of training and validation accuracy over epochs.

     Parameters:
        - model_class (class): The model class to be used (e.g., ResNet, VIT).
        - pretrained (bool, optional): Whether to use a pretrained model. Default is False.
        - learning_rate (float, optional): The learning rate for the optimizer. Default is 0.001.
        - loss_function (torch.nn.Module, optional): The loss function to be used for training. Default is nn.CrossEntropyLoss()
    """
        
    def __init__( self, model_class, pretrained = False, learning_rate=0.001, loss_function=nn.CrossEntropyLoss()):
        super().__init__()
        #self.save_hyperparameters()     
        self.learning_rate = learning_rate
        
        self.train_correct = 0  
        self.train_total = 0
        self.val_correct = 0  
        self.val_total = 0
        self.val_accuracies = []

        if pretrained:
            #self.model = model_class(pretrained=True)
            self.model = model_class()
        else:
            self.model = model_class(loss_function)
                                    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
        logits = self(data)
        loss = self.model.loss(logits, target)
        
        if (self.current_epoch + 1) % 5 == 0:
            preds = logits.argmax(dim=1)
            self.train_correct += (preds == target).sum().item()
            self.train_total += target.size(0)
        
        return loss


    def validation_step(self, batch, batch_idx):
        data, target = batch
        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
        logits = self(data)
        loss = self.model.loss(logits, target)

        if (self.current_epoch + 1) % 5 == 0:
            preds = logits.argmax(dim=1)
            self.val_correct += (preds == target).sum().item()
            self.val_total += target.size(0)

        return loss
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            train_accuracy = self.train_correct / self.train_total
            print(f"Epoch {self.current_epoch + 1} - Training Accuracy: {train_accuracy * 100:.2f}%")
            self.train_correct = 0
            self.train_total = 0

            val_accuracy = self.val_correct / self.val_total
            self.val_accuracies.append(val_accuracy)
            print(f"Epoch {self.current_epoch + 1} - Validation Accuracy: {val_accuracy * 100:.2f}%")
            self.val_correct = 0
            self.val_total = 0


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Zníženie LR každých 10 epôch
        return [optimizer], [scheduler]

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    runs = 5
    dataset_mapping = get_dataset_mapping()
    for dataset_name, type,  model_class, num_classes, pretrained in dataset_mapping:
        start_time = time.time()
        model_name = model_class.__name__
        subprocess.run(['mkdir', '-p', f"{my_path()}/networks/{dataset_name}_{model_name}"])
        
        #for graph later on
        val_accuracies_all_runs = []
        
        # Load dataset
        train_loader, test_loader, _, classes = load_data(dataset=dataset_name, batch_size_train=64, batch_size_test=512)
        
        for run in range(runs):
            print(f"Run {run + 1}/{runs} for {dataset_name} using {model_name}. Device: {'GPU' if device == 'gpu' else 'CPU'}")
            # Initialize model
            model = Model(model_class=model_class, pretrained=pretrained)

            #inicialize epochs
            epoch = 25
            if dataset_name == 'CIFAR10' or dataset_name == 'SVHN':
                epoch = 40
            
            # Define trainer
            trainer = Trainer(
                max_epochs=epoch,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                barebones=True,
                enable_checkpointing=False,
                logger=False,
                check_val_every_n_epoch=1,
                enable_progress_bar=False,
            )
            # Train model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
            
            # Save model
            torch.save(model.state_dict(), my_path()+ f'/networks/{dataset_name}_{model_name}/{dataset_name}_run_{run}.pth')

            #save accuracies for graph later
            val_accuracies_all_runs.append(model.val_accuracies)

        run_time = time.time() - start_time
        print(f"Run {run + 1}/5 for {dataset_name} completed in {run_time:.2f} seconds.")
        
        val_accuracies_all_runs = np.array(val_accuracies_all_runs)
        compute_dev_plot(n_epochs= epoch,data=val_accuracies_all_runs,  name=  my_path()+ f'/networks/{dataset_name}_{model_name}/{dataset_name}.png', title=f'Testing Accuracy for {dataset_name} with {type}')

