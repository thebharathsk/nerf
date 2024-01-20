import os
import argparse
import yaml
import torch
import wandb
import cv2 as cv2
import numpy as np

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss
from metrics import get_metrics
from utils.utils import sampler_coarse, sampler_fine, render, get_time_string, create_log_files

#set wandb entity
os.environ['WANDB_ENTITY'] = 'bsomayaj'

class NeRFEngine(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        #store config
        self.config = config
        
        #create log file
        self.log_file = create_log_files(self.config['exp']['dir'])[0]
        
        #initialize embeddings
        self.embeddings_coarse = get_embeddings(config)
        self.embeddings_fine = get_embeddings(config)
        
        #initialize coarse and fine models
        self.model_coarse = get_model(config)
        self.model_fine = get_model(config)
        
        #initialize loss function
        self.loss_fn_coarse = get_loss(config, tag='coarse')
        self.loss_fn_fine = get_loss(config, tag='fine')
        
        #log begging of training
        self.log_file.info(f'Starting experiment with id {self.config["exp"]["id"]}')
        self.log_file.info(f'PID = {os.getpid()}')
        print(f'PID = {os.getpid()}')
        
    def training_step(self, batch, batch_idx):
        #STEP 1: forward pass through coarse model
        #sample along rays
        samples_coarse = sampler_coarse(batch, self.config['hyperparams']['num_samples_coarse'])
        
        #get embeddings
        embeddings_coarse = self.embeddings_coarse(samples_coarse)
        
        #pass through model
        outputs_coarse = self.model_coarse(embeddings_coarse)
        
        #get rendered colors
        renders_coarse = render(samples_coarse, outputs_coarse, add_sigma_noise=True)
        
        #compute loss
        loss_coarse = self.loss_fn_coarse(batch, samples_coarse, renders_coarse)
        
        #STEP 2: forward pass through fine model
        #sample along rays
        samples_fine = sampler_fine(batch, samples_coarse, renders_coarse, self.config['hyperparams']['num_samples_fine'])
        
        #get embeddings
        embeddings_fine = self.embeddings_fine(samples_fine)
        
        #pass through model
        outputs_fine = self.model_fine(embeddings_fine)
        
        #get rendered colors
        renders_fine = render(samples_fine, outputs_fine, add_sigma_noise=True)
        
        #compute loss
        loss_fine = self.loss_fn_fine(batch, samples_fine, renders_fine)
        
        #concatenate loss
        loss_dict = {}
        loss_dict.update(loss_coarse)
        loss_dict.update(loss_fine)
        
        #add loss
        loss_total = sum(loss_dict.values())
        
        #add prefix to loss dictionary
        loss_dict = {f"train/{key}": value for key, value in loss_dict.items()}
        
        #log loss
        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_total", loss_total, prog_bar=True, on_step=True, on_epoch=False)
        
        # log learning rate
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss_total
    
    # save checkpoints at the end of epochs
    def on_train_epoch_end(self):
        # get the path
        save_path_coarse = os.path.join(self.config['exp']['dir'], \
                                 'checkpoint_latest_coarse.pth')
        save_path_fine = os.path.join(self.config['exp']['dir'], \
                                 'checkpoint_latest_fine.pth')
        
        #save models
        torch.save(self.model_coarse.state_dict(), save_path_coarse)
        torch.save(self.model_fine.state_dict(), save_path_fine)

        # log at end of epoch
        self.log_file.info(f'Epoch {self.current_epoch} completed')
        
    def on_validation_epoch_start(self):
        #shape of image
        num_cameras = self.trainer.val_dataloaders.dataset.num_cameras
        h = self.trainer.val_dataloaders.dataset.h
        w = self.trainer.val_dataloaders.dataset.w
        
        #create arrays to hold images
        self.val_reconstructed_image = torch.zeros((num_cameras, h, w, 3)).float().to(self.device)
        self.val_reconstructed_depth = torch.zeros((num_cameras, h, w)).float().to(self.device)
        self.val_reconstructed_accumulation = torch.zeros((num_cameras, h, w)).float().to(self.device)
        
        #set variables to not require gradients
        self.val_reconstructed_image.requires_grad = False
        self.val_reconstructed_depth.requires_grad = False
        self.val_reconstructed_accumulation.requires_grad = False
    
    def validation_step(self, batch, batch_idx):
        #STEP 1: forward pass through coarse model
        #sample along rays
        samples_coarse = sampler_coarse(batch, self.config['hyperparams']['num_samples_coarse'])
        
        #get embeddings
        embeddings_coarse = self.embeddings_coarse(samples_coarse)
        
        #pass through model
        outputs_coarse = self.model_coarse(embeddings_coarse)
        
        #get rendered colors
        renders_coarse = render(samples_coarse, outputs_coarse)
        
        #compute loss
        loss_coarse = self.loss_fn_coarse(batch, samples_coarse, renders_coarse)
        
        #STEP 2: forward pass through fine model
        #sample along rays
        samples_fine = sampler_fine(batch, samples_coarse, renders_coarse, self.config['hyperparams']['num_samples_fine'])
        
        #get embeddings
        embeddings_fine = self.embeddings_fine(samples_fine)
        
        #pass through model
        outputs_fine = self.model_fine(embeddings_fine)
        
        #get rendered colors
        renders_fine = render(samples_fine, outputs_fine)
        
        #compute loss
        loss_fine = self.loss_fn_fine(batch, samples_fine, renders_fine)
        
        #concatenate loss
        loss_dict = {}
        loss_dict.update(loss_coarse)
        loss_dict.update(loss_fine)
        
        #add loss
        loss_total = sum(loss_dict.values())
        
        #add prefix to loss dictionary
        loss_dict = {f"val/{key}": value for key, value in loss_dict.items()}
        
        #log loss
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/loss_total", loss_total, prog_bar=False, on_step=False, on_epoch=True)
                        
        #transfer colors and depths to image
        batch_ids = batch['ray_id']
        self.val_reconstructed_image[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = renders_fine['rgb']
        self.val_reconstructed_depth[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = renders_fine['depth']
        self.val_reconstructed_accumulation[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = renders_fine['acc']
        
        return loss_total

    def on_validation_epoch_end(self):        
        #save image
        img_np = self.val_reconstructed_image.detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = img_np[...,::-1]
        img_np = np.uint8(img_np*255)
        for img_num in range(img_np.shape[0]):
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'train_{img_num}.png'), img_np[img_num])
        
        #save depth
        depth_np = self.val_reconstructed_depth.detach().cpu().numpy()
        depth_np = (depth_np - np.min(depth_np))/(np.max(depth_np) - np.min(depth_np))
        depth_np = np.uint8(depth_np*255)
        for depth_num in range(depth_np.shape[0]):
            depth_np_colormaped = cv2.applyColorMap(depth_np[depth_num], cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'train_depth_{depth_num}.png'), depth_np_colormaped)
        
        #save accumulation
        acc_np = self.val_reconstructed_accumulation.detach().cpu().numpy()
        acc_np = (acc_np - np.min(acc_np))/(np.max(acc_np) - np.min(acc_np))
        acc_np = np.uint8(acc_np*255)
        for acc_num in range(acc_np.shape[0]):
            acc_np_colormaped = cv2.applyColorMap(acc_np[acc_num], cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'train_acc_{acc_num}.png'), acc_np_colormaped)

    def test_step(self, batch, batch_idx):
        #STEP 1: forward pass through coarse model
        #sample along rays
        samples_coarse = sampler_coarse(batch, self.config['hyperparams']['num_samples_coarse'])
        
        #get embeddings
        embeddings_coarse = self.embeddings_coarse(samples_coarse)
        
        #pass through model
        outputs_coarse = self.model_coarse(embeddings_coarse)
        
        #get rendered colors
        renders_coarse = render(samples_coarse, outputs_coarse)
        
        #compute loss
        loss_coarse = self.loss_fn_coarse(batch, samples_coarse, renders_coarse)
        
        #STEP 2: forward pass through fine model
        #sample along rays
        samples_fine = sampler_fine(batch, samples_coarse, renders_coarse, self.config['hyperparams']['num_samples_fine'])
        
        #get embeddings
        embeddings_fine = self.embeddings_fine(samples_fine)
        
        #pass through model
        outputs_fine = self.model_fine(embeddings_fine)
        
        #get rendered colors
        renders_fine = render(samples_fine, outputs_fine)
        
        #compute loss
        loss_fine = self.loss_fn_fine(batch, samples_fine, renders_fine)
        
        #concatenate loss
        loss_dict = {}
        loss_dict.update(loss_coarse)
        loss_dict.update(loss_fine)
        
        #add loss
        loss_total = sum(loss_dict.values())
        
        #add prefix to loss dictionary
        loss_dict = {f"test/{key}": value for key, value in loss_dict.items()}
        
        #log loss
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/loss_total", loss_total, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss_total
    
    def configure_optimizers(self):
        if self.config['optimizer']['name'] == "adam":
            optimizer = optim.Adam(list(self.model_coarse.parameters()) + list(self.model_fine.parameters()), \
                                lr=self.config['hyperparams']['lr'],
                                betas=(0.9, 0.999))
            
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.91)
        
        return [optimizer], [lr_scheduler]
                
#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """    
    #initialize dataloaders
    train_dataloader = get_dataloader('train', config)
    val_dataloader = get_dataloader('val', config)
    test_dataloader = get_dataloader('test', config)
    
    # create root folder
    exp_id = get_time_string()
    exp_dir = os.path.join(config['exp']['root'], 'exps', exp_id + '_' + config['exp']['name'])
    config['exp']['id'] = exp_id
    config['exp']['dir'] = exp_dir
    wandb_path = os.path.join(exp_dir, 'wandb')
    os.makedirs(wandb_path, exist_ok=True)
    
    #define logger
    wandb_logger = WandbLogger(name=config['exp']['name'],
                                save_dir=wandb_path,
                                project="nerf_simple",
                                offline=config['exp']['offline_logging'])
    
    # log hyperparameters
    wandb_logger.log_hyperparams(config)
    
    # log training script
    wandb_run = wandb_logger.experiment
    artifact = wandb.Artifact('train_script', type='code')
    artifact.add_file(os.path.join(config['exp']['root'], 'src', 'train.py'))
    wandb_run.log_artifact(artifact)
    
    #initialize lightning module
    nerf_engine = NeRFEngine(config)
    
    #initialize trainer
    trainer = L.Trainer(max_epochs=config['hyperparams']['epochs'],
                         accelerator='cuda', 
                         devices=[0], 
                         precision='16',
                         logger=wandb_logger
                        )
    
    #start training
    trainer.fit(nerf_engine, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    #test at the end
    trainer.test(nerf_engine, dataloaders=test_dataloader)
    

if __name__ == "__main__":
    #set seed
    L.seed_everything(27011996)
    
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    #run the code
    main(config)
    
    #indicate end of program
    os.system('play -nq -t alsa synth 0.05 sine 440')
    
    