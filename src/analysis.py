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
        #log image
        image_pil = self.test_reconstructed_image.squeeze(0).permute(1,2,0)
        image_pil = image_pil.cpu().numpy()*255
        image_pil = image_pil.astype('uint8')
        image_pil = Image.fromarray(image_pil[:,:,::-1])
        
        #save image
        image_pil.save(os.path.join(self.logger.save_dir, 'analysis_reconstructed.png'))
                
#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    analysis_dataloader = get_dataloader('analysis', config['data']['analysis'], 
                                     config['hyperparams'])
    
    #initialize lightning module
    nerf_engine = NeRFEngine.load_from_checkpoint(config['paths']['resume'], strict=False, config=config)
    
    #initialize trainer
    trainer = L.Trainer(accelerator='cuda', 
                         devices=[0], 
                         precision='16')
    
    #test on dataloader
    trainer.test(nerf_engine, dataloaders=analysis_dataloader)

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)