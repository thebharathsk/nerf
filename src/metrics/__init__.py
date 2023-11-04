from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics import MeanSquaredError as mse

from torchmetrics import MetricCollection

def get_metrics(metric_args, prefix):
    """Get dictionary of metrics

    Args:
        metric_args: arguments for metrics
        prefix: prefix for metrics
    """
    #populate metrics dictionary
    metrics_dict = {}
    #add metrics
    if 'train' not in prefix:
        if 'psnr' in metric_args['list']:
            metrics_dict['psnr'] = psnr(data_range=(0,1.0))
        
        if 'lpips' in metric_args['list']:
            metrics_dict['lpips'] = lpips()
        
        if 'ssim' in metric_args['list']:
            metrics_dict['ssim'] = ssim(data_range=(0,1.0))
    
    #use only MSE for train
    else:
        metrics_dict['mse'] = mse()
    
    #create mertic collection
    metrics = MetricCollection(metrics=metrics_dict, prefix=prefix+'/')
    
    return metrics