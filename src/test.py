import argparse
import yaml
import torch

import numpy as np
import cv2 as cv2
import torch.optim as optim
from tqdm import tqdm

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss
from metrics import get_metrics
from utils.utils import sampler, render

#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    train_dataloader = get_dataloader('train', config)
    val_dataloader = get_dataloader('val', config)
    
    #initialize embeddings
    embeddings = get_embeddings(config).to('cuda')
    
    #initialize model
    model = get_model(config).to('cuda')
    model.train()
    
    #initialize loss function
    loss_fn = get_loss(config).to('cuda')

    #initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparams']['lr'])

    #empty image
    img = torch.zeros((1080,1920,3)).to('cuda').float()
    
    #get a batch from dataloader
    for i, batch in enumerate(train_dataloader):
        if i > 500:
            break
        #sample along rays
        locs, dirs, t_sampled = sampler(batch, config['hyperparams']['num_samples_coarse'], fine=False)
        
        #shift locs, dirs, t_sampled to cuda
        locs = locs.to('cuda')
        dirs = dirs.to('cuda')
        t_sampled = t_sampled.to('cuda')
        
        #get embeddings
        locs_emb, dirs_emb = embeddings(locs, dirs)
        
        #pass through model
        sigma, rgb = model(locs_emb, dirs_emb)
        
        #get rendered colors
        rgb_rendered = render(rgb, sigma, t_sampled)
        
        #compute loss
        loss = loss_fn(rgb_rendered, batch['ray_rgb'].to('cuda'))
        
        #backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print loss
        if i % 10 == 0:        
            print(f"iter = {i}, loss = {loss}")
        
        #validate
        if i>0 and i%100 == 0:
            #switch to evaluation mode
            model.eval()
            
            for i, batch_val in tqdm(enumerate(val_dataloader)):
                #sample along rays
                locs, dirs, t_sampled = sampler(batch_val, config['hyperparams']['num_samples_coarse'], fine=False)
                
                #shift locs, dirs, t_sampled to cuda
                locs = locs.to('cuda')
                dirs = dirs.to('cuda')
                t_sampled = t_sampled.to('cuda')
                
                #get embeddings
                locs_emb, dirs_emb = embeddings(locs, dirs)
                
                print(locs_emb.shape, dirs_emb.shape)
                
                #pass through model
                sigma, rgb = model(locs_emb, dirs_emb)
                
                #get rendered colors
                rgb_rendered = render(rgb, sigma, t_sampled)
                
                #add colors to image
                yxs = batch_val['ray_id'][:,1:]
                img[yxs[:,0], yxs[:,1]] = rgb_rendered
            
            #save image
            img_np = img.detach().cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_np = img_np[:,:,::-1]
            img_np = np.uint8(img_np*255)
            cv2.imwrite('test.png', img_np)
            
            #reset to trianing mode
            model.train()
    
    # for i, batch in enumerate(train_dataloader):
    #     if i > 5:
    #         break
    #     print(batch['ray_id'][0])
    #     print(batch['ray_o'][0], batch['ray_d'][0])        
    #     print(batch['ray_o'][0] + batch['ray_d'][0]*batch['ray_bds'][0,0])
    #     print(batch['ray_o'][0] + batch['ray_d'][0]*batch['ray_bds'][0,1])
    #     print('############')
    # print(len(train_dataloader)*config['hyperparams']['num_rays'], 126*1920*1080)
    
if __name__ == "__main__":
    #set numpy as torch seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)