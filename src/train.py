import os
import argparse
import lightning as pl

class NeRFEngine(pl.LightningModule):
    def __init__(self, args):
        """Constructor for NeRF Engine

        Args:
            args: arguments from command line
        """
        super().__init__()
        self.args = args
        
        #initialize model

        #initialize loss functions

#main function
def main(args):
    """Main function
    
        Args:
            args: arguments from command line
    """
    #initialize lightning module
    
    #create training options
    
    #train model
    
    #evaluate model
    

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to image')
    parser.add_a

    args = parser.parse_args()

    main(args)