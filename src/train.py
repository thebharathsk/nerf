import os
import argparse
import yaml
import torch
from PIL import Image

import lightning as L
from lightning.pytorch.loggers import CometLogger
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss
from metrics import get_metrics

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
        
        #initialize metrics
        self.train_metrics = get_metrics(config['metrics'], prefix='train')
        self.val_metrics = get_metrics(config['metrics'], prefix='val')
        self.test_metrics = get_metrics(config['metrics'], prefix='test')
    
    def on_train_start(self):
        #log hyperparameters
        self.logger.experiment.log_parameters(self.config)
        
    def training_step(self, batch, batch_idx):
        #unpack batch
        coords, colors = batch['coords'], batch['colors']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
        
        #compute loss
        loss = self.loss(colors_pred, colors)
        
        #compute metrics
        metrics = self.train_metrics(colors_pred, colors)
        
        #log loss
        self.log('train_loss', loss, prog_bar=True)
        
        #log other metrics
        self.log_dict(metrics)
        
        return loss
        
    def on_validation_epoch_start(self):
        #create arrays to hold images
        self.val_reconstructed_image = torch.zeros(self.trainer.val_dataloaders.dataset.image.shape).float().to(self.device)
        self.val_original_image = torch.zeros(self.trainer.val_dataloaders.dataset.image.shape).float().to(self.device)
        
        #convert to NCHW format
        self.val_reconstructed_image = self.val_reconstructed_image.permute(2,0,1).unsqueeze(0)
        self.val_original_image = self.val_original_image.permute(2,0,1).unsqueeze(0)
        
        #set variables to not require gradients
        self.val_reconstructed_image.requires_grad = False
        self.val_original_image.requires_grad = False
    
    def validation_step(self, batch, batch_idx):
        #unpack batch
        coords, colors, coords_unnorm = batch['coords'], batch['colors'], batch['coords_unnorm']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
        
        #reshape colors
        colors = colors.permute(1,0).unsqueeze(0).float()
        colors_pred = colors_pred.permute(1,0).unsqueeze(0).float()
                
        #add colors to images
        self.val_reconstructed_image[:,:,coords_unnorm[:,1], coords_unnorm[:,0]] = colors_pred
        self.val_original_image[:,:,coords_unnorm[:,1], coords_unnorm[:,0]] = colors
    
    def on_validation_epoch_end(self):
        #compute metrics
        metrics = self.val_metrics(self.val_reconstructed_image, self.val_original_image)
        
        #log metrics
        self.log_dict(metrics)
        
        #log image
        image_pil = self.val_reconstructed_image.squeeze(0).permute(1,2,0)
        image_pil = image_pil.cpu().numpy()*255
        image_pil = image_pil.astype('uint8')
        image_pil = Image.fromarray(image_pil[:,:,::-1])
        self.logger.experiment.log_image(image_pil, name='val_reconstructed_image')
        
        #save image
        image_pil.save(os.path.join(self.logger.save_dir, 'val_reconstructed.png'))
    
    def on_test_epoch_start(self):
        #create arrays to hold images
        self.test_reconstructed_image = torch.zeros(self.trainer.test_dataloaders.dataset.image.shape).float().to(self.device)
        self.test_original_image = torch.zeros(self.trainer.test_dataloaders.dataset.image.shape).float().to(self.device)
    
        #convert to NCHW format
        self.test_reconstructed_image = self.test_reconstructed_image.permute(2,0,1).unsqueeze(0)
        self.test_original_image = self.test_original_image.permute(2,0,1).unsqueeze(0)
        
        #set variables to not require gradients
        self.test_reconstructed_image.requires_grad = False
        self.test_original_image.requires_grad = False

    def test_step(self, batch, batch_idx):
        #unpack batch
        coords, colors, coords_unnorm = batch['coords'], batch['colors'], batch['coords_unnorm']
        
        #get embeddings
        embeddings = self.embeddings(coords)
        
        #get outputs
        colors_pred = self.model(embeddings)
        
        #reshape colors
        colors = colors.permute(1,0).unsqueeze(0).float()
        colors_pred = colors_pred.permute(1,0).unsqueeze(0).float()
        
        #add colors to images
        self.test_reconstructed_image[:,:,coords_unnorm[:,1], coords_unnorm[:,0]] = colors_pred
        self.test_original_image[:,:,coords_unnorm[:,1], coords_unnorm[:,0]] = colors
        
    def on_test_epoch_end(self):
        #compute metrics
        metrics = self.test_metrics(self.test_reconstructed_image, self.test_original_image)
        
        #log metrics
        self.log_dict(metrics)
        
        #log image
        image_pil = self.test_reconstructed_image.squeeze(0).permute(1,2,0)
        image_pil = image_pil.cpu().numpy()*255
        image_pil = image_pil.astype('uint8')
        image_pil = Image.fromarray(image_pil[:,:,::-1])
        self.logger.experiment.log_image(image_pil, name='test_reconstructed_image')
        
        #save image
        image_pil.save(os.path.join(self.logger.save_dir, 'test_reconstructed.png'))
    
    def configure_optimizers(self):
        if self.config['optimizer']['name'] == "adam":
            optimizer = optim.Adam(self.model.parameters(), \
                                lr=self.config['hyperparams']['lr'],
                                weight_decay=self.config['optimizer']['weight_decay'])
        
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        
        return [optimizer], [lr_scheduler]
                
#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    train_dataloader = get_dataloader('train',
                                      config['data']['train'], 
                                      config['hyperparams'])
    val_dataloader = get_dataloader('val',
                                    config['data']['val'], 
                                    config['hyperparams'])
    test_dataloader = get_dataloader('test',
                                     config['data']['test'], 
                                     config['hyperparams'])
    
    #define logger
    comet_logger = CometLogger(project_name="nerf-image",
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
                         strategy='ddp_find_unused_parameters_true')
    
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