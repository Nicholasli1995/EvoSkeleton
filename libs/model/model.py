"""
Fully-connected residual network as a single deep learner.
"""

import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    A residual block.
    """
    def __init__(self, linear_size, p_dropout=0.5, kaiming=False, leaky=False):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        
        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCModel(nn.Module):
    def __init__(self,
                 stage_id=1,
                 linear_size=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=False,
                 refine_3d=False, 
                 leaky=False,
                 dm=False,
                 input_size=32,
                 output_size=64):
        """
        Fully-connected network.
        """
        super(FCModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.stage_id = stage_id
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm 
        self.input_size = input_size
        if self.stage_id>1 and self.refine_3d:
            self.input_size += 16 * 3        
        # 3d joints
        self.output_size = output_size

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.res_blocks = []
        for l in range(num_blocks):
            self.res_blocks.append(ResidualBlock(self.linear_size, 
                                                 self.p_dropout,
                                                 leaky=self.leaky))
        self.res_blocks = nn.ModuleList(self.res_blocks)
        
        # output
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.get_representation(x)
        y = self.w2(y)
        return y
    
    def get_representation(self, x):
        # get the latent representation of an input vector
        # first layer
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # residual blocks
        for i in range(self.num_blocks):
            y = self.res_blocks[i](y)        
        
        return y
    
def get_model(stage_id, 
              refine_3d=False, 
              norm_twoD=False, 
              num_blocks=2,
              input_size=32, 
              output_size=64, 
              linear_size=1024, 
              dropout=0.5,
              leaky=False
              ):
    model = FCModel(stage_id=stage_id, 
                    refine_3d=refine_3d, 
                    norm_twoD=norm_twoD,
                    num_blocks=num_blocks, 
                    input_size=input_size, 
                    output_size=output_size, 
                    linear_size=linear_size,
                    p_dropout=dropout, 
                    leaky=leaky
                    )
    return model

def prepare_optim(model, opt):
    """
    Prepare optimizer.
    """
    params = [ p for p in model.parameters() if p.requires_grad]
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params, 
                                     lr = opt.lr, 
                                     weight_decay = opt.weight_decay
                                     )
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params, 
                                    lr = opt.lr, 
                                    momentum = opt.momentum,
                                    weight_decay = opt.weight_decay
                                    )    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones = opt.milestones, 
                                                     gamma = opt.gamma)
    return optimizer, scheduler

def get_cascade():
    """
    Get an empty cascade.
    """
    return nn.ModuleList([])