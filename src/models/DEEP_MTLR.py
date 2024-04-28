import torch
import torch.nn as nn
from models.components.swin import SwinUNETR
from models.components.net import UNETR, ViT



class DEEP_MTLR(torch.nn.Module):


    def __init__(self,hparams, model_type: str = 'UNETR'):
        super(DEEP_MTLR, self).__init__()

        if model_type == 'UNETR':
            feature_size = 8
            self.model = UNETR( 
                                    hparams = hparams,
                                    in_channels= 2,
                                    out_channels= 1,
                                    img_size = (96, 96, 96), #change
                                    feature_size = feature_size,
                                    hidden_size = 768,
                                    mlp_dim = 3072,
                                    num_heads = 12,
                                    pos_embed = "conv",
                                    norm_name = "instance",
                                    conv_block = True,
                                    res_block = True,
                                    dropout_rate = 0.0,
                                    spatial_dims = 3,
                            )
        elif model_type == 'ViT':
            self.model = ViT(   
                                    in_channels=2,
                                    img_size=(96, 96, 96), #change
                                    patch_size=3,
                                    hidden_size=768,
                                    mlp_dim=3072,
                                    num_heads=12,
                                    pos_embed="conv",
                                    classification=False,
                                    dropout_rate=0.0,
                                    spatial_dims=3,
                            )
        elif model_type == 'SWIN':
            feature_size = hparams['featureSize']
            self.model = SwinUNETR( 
                                    hparams = hparams,
                                    img_size = (96, 96, 96),
                                    in_channels = 2,
                                    out_channels = 1,
                                    depths = (2, 2, 2, 2),
                                    num_heads = (3, 6, 12, 24),
                                    feature_size = feature_size,
                                    norm_name = "instance",
                                    drop_rate = 0.0,
                                    attn_drop_rate = 0.0,
                                    dropout_path_rate = 0.0,
                                    normalize = True,
                                    use_checkpoint = False, 
                                    spatial_dims = 3,
                                    downsample="merging",
                                    use_v2=False,      
                                )
        else:
            print('Please select the correct model architecture name.')

        self.init_params(self.model)   


    def forward(self, x):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

            nn.init.constant_(m.bias, -1.5214691)

