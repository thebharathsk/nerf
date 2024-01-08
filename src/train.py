import os
import argparse
import yaml
import torch
import cv2 as cv2
import numpy as np

import lightning as L
from lightning.pytorch.loggers import CometLogger
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss
from metrics import get_metrics
from utils.utils import sampler, render

class NeRFEngine(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #initialize embeddings
        self.embeddings = get_embeddings(config)
        
        #initialize model
        self.model = get_model(config)
        
        #initialize loss function
        self.loss_fn = get_loss(config)
    
    def on_train_start(self):
        #log hyperparameters
        self.logger.experiment.log_parameters(self.config)
        
    def training_step(self, batch, batch_idx):
        #sample along rays
        locs, dirs, t_sampled = sampler(batch, self.config['hyperparams']['num_samples_coarse'], fine=False)
        
        #get embeddings
        locs_emb, dirs_emb = self.embeddings(locs, dirs)
        
        #pass through model
        sigma, rgb = self.model(locs_emb, dirs_emb)
        
        #get rendered colors
        rgb_rendered, _ = render(rgb, sigma, t_sampled)
        
        #compute loss
        loss = self.loss_fn(rgb_rendered, batch['ray_rgb'])
        
        #log loss
        self.log('train/loss', loss, prog_bar=True)
        
        # log learning rate
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return loss
        
    def on_validation_epoch_start(self):
        #shape of image
        num_cameras = self.trainer.val_dataloaders.dataset.num_cameras
        h = self.trainer.val_dataloaders.dataset.h
        w = self.trainer.val_dataloaders.dataset.w
        
        #create arrays to hold images
        self.val_reconstructed_image = torch.zeros((num_cameras, h, w, 3)).float().to(self.device)
        self.val_reconstructed_depth = torch.zeros((num_cameras, h, w)).float().to(self.device)
        
        #set variables to not require gradients
        self.val_reconstructed_image.requires_grad = False
        self.val_reconstructed_depth.requires_grad = False
    
    def validation_step(self, batch, batch_idx):
        #sample along rays
        locs, dirs, t_sampled = sampler(batch, self.config['hyperparams']['num_samples_coarse'], fine=False)
        
        #get embeddings
        locs_emb, dirs_emb = self.embeddings(locs, dirs)
        
        #pass through model
        sigma, rgb = self.model(locs_emb, dirs_emb)
        
        #get rendered colors
        rgb_rendered, depth_rendered = render(rgb, sigma, t_sampled)
        
        #compute loss
        loss = self.loss_fn(rgb_rendered, batch['ray_rgb'])
        
        #log loss
        self.log('val/loss', loss, on_epoch=True, on_step=False)
        
        #transfer colors and depths to image
        batch_ids = batch['ray_id']
        self.val_reconstructed_image[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = rgb_rendered
        self.val_reconstructed_depth[batch_ids[:,0], batch_ids[:,1], batch_ids[:,2]] = depth_rendered
        
        return loss

    def on_validation_epoch_end(self):        
        #save image
        img_np = self.val_reconstructed_image.detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = img_np[...,::-1]
        img_np = np.uint8(img_np*255)
        for img_num in range(img_np.shape[0]):
            cv2.imwrite(f'test_{img_num}.png', img_np[img_num])
        
        #save depth
        depth_np = self.val_reconstructed_depth.detach().cpu().numpy()
        depth_np = (depth_np - np.min(depth_np))/(np.max(depth_np) - np.min(depth_np))
        depth_np = np.uint8(depth_np*255)
        for depth_num in range(depth_np.shape[0]):
            cv2.imwrite(f'test_depth_{depth_num}.png', depth_np[depth_num])

    # def on_test_epoch_start(self):
    #     #create arrays to hold images
    #     self.test_reconstructed_image = torch.zeros(self.trainer.test_dataloaders.dataset.image.shape).float().to(self.device)
    #     self.test_original_image = torch.zeros(self.trainer.test_dataloaders.dataset.image.shape).float().to(self.device)
    
    #     #convert to NCHW format
    #     self.test_reconstructed_image = self.test_reconstructed_image.permute(2,0,1).unsqueeze(0)
    #     self.test_original_image = self.test_original_image.permute(2,0,1).unsqueeze(0)
        
    #     #set variables to not require gradients
    #     self.test_reconstructed_image.requires_grad = False
    #     self.test_original_image.requires_grad = False

    def test_step(self, batch, batch_idx):
        #sample along rays
        locs, dirs, t_sampled = sampler(batch, self.config['hyperparams']['num_samples_coarse'], fine=False)
        
        #get embeddings
        locs_emb, dirs_emb = self.embeddings(locs, dirs)
        
        #pass through model
        sigma, rgb = self.model(locs_emb, dirs_emb)
        
        #get rendered colors
        rgb_rendered, _ = render(rgb, sigma, t_sampled)
        
        #compute loss
        loss = self.loss_fn(rgb_rendered, batch['ray_rgb'])
        
        #log loss
        self.log('test/loss', loss, on_epoch=True, on_step=False)
        
        return loss
        
    # def on_test_epoch_end(self):
    #     #compute metrics
    #     metrics = self.test_metrics(self.test_reconstructed_image, self.test_original_image)
        
    #     #log metrics
    #     self.log_dict(metrics)
        
    #     #log image
    #     image_pil = self.test_reconstructed_image.squeeze(0).permute(1,2,0)
    #     image_pil = image_pil.cpu().numpy()*255
    #     image_pil = image_pil.astype('uint8')
    #     image_pil = Image.fromarray(image_pil[:,:,::-1])
    #     self.logger.experiment.log_image(image_pil, name='test_reconstructed_image')
        
    #     #save image
    #     image_pil.save(os.path.join(self.logger.save_dir, 'test_reconstructed.png'))
    
    def configure_optimizers(self):
        if self.config['optimizer']['name'] == "adam":
            optimizer = optim.Adam(self.model.parameters(), \
                                lr=self.config['hyperparams']['lr'],
                                betas=(0.9, 0.999))
        # lr_scheduler = {
        #         'scheduler': StepLR(optimizer, step_size=1, gamma=0.99),
        #         'interval': 'epoch',
        #         'frequency': 1
        #         }
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        
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
    
    #define logger
    save_dir = os.path.join(config['exp']['root'], config['exp']['name'])
    os.makedirs(save_dir, exist_ok=True)
    comet_logger = CometLogger(project_name="nerf-colmap",
                               save_dir=save_dir,
                               experiment_name=config['exp']['name'])
    
    #initialize lightning module
    nerf_engine = NeRFEngine(config)
    
    #initialize trainer
    trainer = L.Trainer(max_epochs=config['hyperparams']['epochs'],
                         accelerator='cuda', 
                         devices=[0], 
                         precision='16',
                         logger=comet_logger
                        )
    
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
    
    #run the code
    main(config)
    
    #indicate end of program
    os.system('play -nq -t alsa synth 0.05 sine 440')
    
    