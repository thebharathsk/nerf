import argparse
import yaml

from embeddings import get_embeddings
from models import get_model
from dataloaders import get_dataloader
from loss import get_loss
from metrics import get_metrics
from utils.utils import sampler

#main function
def main(config):
    """Main function
    
        Args:
            config: configuration for training
    """
    #initialize dataloaders
    train_dataloader = get_dataloader('train', config)
    # val_dataloader = get_dataloader('val', config)
    # test_dataloader = get_dataloader('test', config)
    
    #initialize model
    model = get_model(config)
    
    #initialize embeddings
    embeddings = get_embeddings(config)

    #get a batch from dataloader
    batch = next(iter(train_dataloader))
    
    #sample along rays
    locs, dirs = sampler(batch, config['hyperparams']['num_samples_coarse'], fine=False)
    print(locs.shape, dirs.shape)
    
    #get embeddings
    locs_emb, dirs_emb = embeddings(locs, dirs)
    print(locs_emb.shape, dirs_emb.shape)
    
    #pass through model
    sigma, rgb = model(locs_emb, dirs_emb)
    print(sigma.shape, rgb.shape)
    print(sigma.min(), sigma.max())
    print(rgb.min(), rgb.max())
    
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
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to configuation file')
    args = parser.parse_args()
    
    #load configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config)