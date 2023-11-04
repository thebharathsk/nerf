import os
import argparse
import yaml
import lightning as L
from lightning.pytorch.loggers import CometLogger
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss

class NeRFEngine(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #initialize embeddings
        self.embeddings = get_embeddings(config['embeddings'])
        
        #initialize model
        self.model = get_model(config['model'])
        
        #initialize loss function
        self.loss = get_loss(config['loss'])

    def training_step(self, batch, batch_idx):
        #unpack batch
        coords, colors = batch['coords'], batch['colors']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
        
        #compute loss
        loss = self.loss(colors_pred, colors)
        
        #log loss
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        #unpack batch
        coords, colors = batch['coords'], batch['colors']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
    
    def test_step(self, batch, batch_idx):
        #unpack batch
        coords, colors = batch['coords'], batch['colors']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
    
    def configure_optimizers(self):
        if self.config['optimizer']['name'] == "adam":
            optimizer = optim.Adam(self.model.parameters(), \
                                lr=self.config['hyperparams']['lr'])
        
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        
        return [optimizer], [lr_scheduler]
                
#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    train_dataloader = get_dataloader('train', config['data']['train'], 
                                      config['hyperparams'])
    val_dataloader = get_dataloader('val', config['data']['val'], 
                                    config['hyperparams'])
    test_dataloader = get_dataloader('test', config['data']['test'], 
                                     config['hyperparams'])
    
    #define logger
    comet_logger = CometLogger(project_name="nerf/image",
                               save_dir=os.path.join(config['paths']['exp_dir'], config['exp_name']),
                               experiment_name=config['exp_name'])
    
    #initialize lightning module
    nerf_engine = NeRFEngine(config)
    
    #initialize trainer
    trainer = L.Trainer(max_epochs=config['hyperparams']['epochs'],
                         accelerator='cuda', 
                         devices=[0,1], 
                         precision='16',
                         logger=comet_logger,
                         log_every_n_steps=50000)
    
    #start training
    trainer.fit(nerf_engine, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    #test at the end
    trainer.test(nerf_engine, dataloaders=test_dataloader)
    

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)