import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
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
        
        #load pretrained weights
        weights_coarse = torch.load(os.path.join(config['exp']['dir'], 'checkpoint_latest_coarse.pth'))
        weights_fine = torch.load(os.path.join(config['exp']['dir'], 'checkpoint_latest_fine.pth'))
        self.model_coarse.load_state_dict(weights_coarse, strict=True)
        self.model_fine.load_state_dict(weights_fine, strict=True)
        
        #initialize loss function
        self.loss_fn_coarse = get_loss(config)
        self.loss_fn_fine = get_loss(config)
        
        #log begging of training
        self.log_file.info(f'Testing experiment with id {self.config["exp"]["name"]}')
        self.log_file.info(f'PID = {os.getpid()}')
        print(f'PID = {os.getpid()}')

    def on_test_epoch_start(self):
        #shape of image
        num_cameras = self.trainer.test_dataloaders.dataset.num_cameras
        h = self.trainer.test_dataloaders.dataset.h
        w = self.trainer.test_dataloaders.dataset.w
        
        #create arrays to hold images
        self.test_reconstructed_image = torch.zeros((num_cameras, h, w, 3)).float().to(self.device)
        self.test_reconstructed_depth = torch.zeros((num_cameras, h, w)).float().to(self.device)
        self.test_reconstructed_accumulation = torch.zeros((num_cameras, h, w)).float().to(self.device)
        
        #set variables to not require gradients
        self.test_reconstructed_image.requires_grad = False
        self.test_reconstructed_depth.requires_grad = False
        self.test_reconstructed_accumulation.requires_grad = False

    def test_step(self, batch, batch_idx):
        #STEP 1: forward pass through coarse model
        #sample along rays
        locs, dirs, t_sampled = sampler_coarse(batch, self.config['hyperparams']['num_samples_coarse'])
        
        #get embeddings
        locs_emb, dirs_emb = self.embeddings_coarse(locs, dirs)
        
        #pass through model
        sigma, rgb = self.model_coarse(locs_emb, dirs_emb)
        
        #get rendered colors
        rgb_rendered, _, _, weights = render(rgb, sigma, t_sampled, batch['ray_d'], locs, plot=False)
        
        #plot weights fine
        weights_np = weights.detach().cpu().numpy()
        t_sampled_np = t_sampled.detach().cpu().numpy()        
        plt.plot(t_sampled_np[0,:,0], weights_np[0], 'b^')
        plt.savefig(os.path.join(self.config['exp']['dir'], f'weights_coarse.png'))
        plt.close()
        
        #compute loss
        loss_coarse = self.loss_fn_coarse(rgb_rendered, batch['ray_rgb'])
        
        #STEP 2: forward pass through fine model
        #sample along rays
        locs_fine, dirs_fine, t_sampled_fine = sampler_fine(batch, t_sampled, weights.detach(), self.config['hyperparams']['num_samples_fine'])
        
        #get embeddings
        locs_emb_fine, dirs_emb_fine = self.embeddings_fine(locs_fine, dirs_fine)
        
        #pass through model
        sigma_fine, rgb_fine = self.model_fine(locs_emb_fine, dirs_emb_fine)
        
        #get rendered colors
        rgb_rendered_fine, depth_rendered_fine, acc_rendered_fine, weights_fine = render(rgb_fine, sigma_fine, t_sampled_fine, batch['ray_d'], locs_fine, plot=False)
        
        #plot weights fine
        weights_fine_np = weights_fine.detach().cpu().numpy()
        t_sampled_fine_np = t_sampled_fine.detach().cpu().numpy()        
        plt.plot(t_sampled_fine_np[0,:,0], weights_fine_np[0], 'r^')
        plt.savefig(os.path.join(self.config['exp']['dir'], f'weights_fine.png'))
        plt.close()

        #compute loss
        loss_fine = self.loss_fn_fine(rgb_rendered_fine, batch['ray_rgb'])
        
        #compute total loss
        loss = loss_coarse + loss_fine
        
        #log loss
        self.log('test/loss_coarse', loss_coarse, on_step=False, on_epoch=True)
        self.log('test/loss_fine', loss_fine, on_step=False, on_epoch=True)
        self.log('test/loss_total', loss, on_step=False, on_epoch=True)
        
        #transfer colors and depths to image
        batch_ids = batch['ray_id']
        self.test_reconstructed_image[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = rgb_rendered_fine
        self.test_reconstructed_depth[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = depth_rendered_fine
        self.test_reconstructed_accumulation[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = acc_rendered_fine
        
        return loss

    def on_test_epoch_end(self):        
        #save image
        img_np = self.test_reconstructed_image.detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = img_np[...,::-1]
        img_np = np.uint8(img_np*255)
        for img_num in range(img_np.shape[0]):
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'eval_{img_num}.png'), img_np[img_num])
        
        #save accumulation
        acc_np = self.test_reconstructed_accumulation.detach().cpu().numpy()
        acc_np = (acc_np - np.min(acc_np))/(np.max(acc_np) - np.min(acc_np))
        acc_np = np.uint8(acc_np*255)
        for acc_num in range(acc_np.shape[0]):
            acc_np_colormaped = cv2.applyColorMap(acc_np[acc_num], cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'eval_acc_{acc_num}.png'), acc_np_colormaped)
            
        #save depth
        depth_np = self.test_reconstructed_depth.detach().cpu().numpy()
        depth_np[acc_np < 220] = np.mean(depth_np)
        depth_np = (depth_np - np.min(depth_np))/(np.max(depth_np) - np.min(depth_np))
        depth_np = np.uint8(depth_np*255)
        for depth_num in range(depth_np.shape[0]):
            depth_np_colormaped = cv2.applyColorMap(depth_np[depth_num], cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.config['exp']['dir'], f'eval_depth_{depth_num}.png'), depth_np_colormaped)
        
             
#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """    
    #initialize dataloaders
    test_dataloader = get_dataloader('test', config)
    
    # create root folder
    exp_dir = os.path.join(config['exp']['root'], config['exp']['name'])
    config['exp']['dir'] = exp_dir
    
    #check if experiment exists
    if not os.path.exists(exp_dir):
        raise ValueError(f'Experiment {exp_dir} does not exist')
    
    #initialize lightning module
    nerf_engine = NeRFEngine(config)
    
    #initialize trainer
    trainer = L.Trainer(max_epochs=0,
                        accelerator='cuda', 
                        devices=[0], 
                        precision='16',
                        logger=False
                        )
    
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