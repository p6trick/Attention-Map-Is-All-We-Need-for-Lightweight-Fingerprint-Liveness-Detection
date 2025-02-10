from opt.base_opt import Opt
from model.model import MyModel, AblationModel, MyinferModel
import torch
import torch.nn as nn
            
            
class Config:
    def __init__(self, opt=Opt):
        self.device = f'cuda:{opt["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
        self.img_size = opt['img_size']
        self.in_chans = opt['in_chans']
        self.depth = opt['depth']
        self.patch_size = opt['patch_size']
        self.embed_dim = opt['embed_dim']
        self.num_heads = opt['num_heads']
        self.batch_size = opt['batch_size']
        self.weight_decay = opt['weight_decay']
        self.betas = opt['betas']
        self.criterion = opt['criterion']
        self.optimizer = opt['optimizer']
        self.scheduler = opt['scheduler']
        self.max_lr = opt['max_lr']
        self.min_lr = opt['min_lr']
        self.gamma = opt['gamma']
        self.epochs = opt['epochs']
        self.num_workers = opt['num_workers']
        self.block_out_list = opt['block_out_list']
        self.down_list = opt['down_list']
        self.block_mid_chan = opt['block_mid_chan']
        self.loss_coef = opt['loss_coef']
        self.transforms = opt['transforms']
        self.use_scheduler = opt['use_scheduler']
        
        
    def load_model(self, model_type='org'):
        if model_type == 'org':
            self.model = MyModel(img_size=self.img_size, in_chans=self.in_chans, 
                    patch_size=self.patch_size, embed_dim=self.embed_dim, 
                    num_heads=self.num_heads, depth=self.depth, block_out_list=self.block_out_list, down_list=self.down_list,block_mid_chan=self.block_mid_chan)
        elif model_type == 'infer':
            self.model = MyinferModel(img_size=self.img_size, in_chans=self.in_chans, 
                    patch_size=self.patch_size, embed_dim=self.embed_dim, 
                    num_heads=self.num_heads, depth=self.depth, block_out_list=self.block_out_list, down_list=self.down_list,block_mid_chan=self.block_mid_chan)
        elif model_type == 'ablation':
            self.model = AblationModel(block_out_list=self.block_out_list, down_list=self.down_list, block_mid_chan=self.block_mid_chan)
        elif model_type == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,2)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = model
        elif model_type == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,2)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model = model
        elif model_type == 'mobilenet':
            import torchvision.models as models
            model = models.mobilenet_v3_small(pretrained=True)
            ftr = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(ftr,2)
            self.model = model
        elif model_type == 'efficientnet':
            import torchvision.models as models

            model = models.efficientnet_v2_s(pretrained=True)
            ftr = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(ftr,2)
            self.model = model
        elif model_type == 'vgg16':
            import torchvision.models as models
            model = models.vgg16(pretrained=True)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=2, bias=True)
            self.model = model
        elif model_type == 'vit-b':
            import timm

            num_classes = 2
            model = timm.create_model('vit_base_patch16_224',img_size=self.img_size, 
                                    pretrained=True,
                                    num_classes=num_classes,
                                    in_chans=1,)
            self.model = model
        
        
        
        self.set_optimizer()
        if self.use_scheduler:
            self.set_scheduler()
        
    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.max_lr, weight_decay=self.weight_decay,betas=self.betas)
        
    def set_scheduler(self):
        if self.scheduler == 'CosineAnnealingWarmupRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)
        elif self.scheduler =='exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
            
#    def load_pretained_model(self,pt_path):
        # pretrained model 불러오는 함수 생성 예정