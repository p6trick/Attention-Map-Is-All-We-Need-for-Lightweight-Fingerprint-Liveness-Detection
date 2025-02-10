from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataloader(config, phase, Data):
    dset = CustomDataset(phase, Data, CustomTransforms(config.transforms))
    if phase == 'train':
        sf = True
    else:
        sf = False
    loader = DataLoader(dset,batch_size=config.batch_size,shuffle=sf,num_workers=config.num_workers)
    return loader

class CustomDataset(Dataset):
    def __init__(self,phase, Data, transforms):
        self.phase = phase
        
        if phase == 'train':
            self.img = Data['x_train']
            self.label = Data['x_target']
        else:
            self.img = Data['x_test']
            self.label = Data['x_label']
        
        self.transforms=transforms
        
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        
        img = self.img[idx]
        
        if self.phase == 'train':
            img = self.transforms(img, self.phase)
        
        label = self.label[idx]
        
        return img, label
    
    
class CustomTransforms(nn.Module):
    def __init__(self, tt):
        t_list = []
        if 'RandomRotate' in tt :
            t_list.append(transforms.RandomRotation(90))
        if 'RandomErasing' in tt:
            t_list.append(
            transforms.Compose([
                    transforms.RandomErasing(p=0.5, scale=(0.01,0.03), value='random'),
                    transforms.RandomErasing(p=0.5, scale=(0.01,0.03), value='random'),
                     transforms.RandomErasing(p=0.5, scale=(0.01,0.03), value='random'),
                 ]))
        self.data_transforms = {
            'train':transforms.Compose(t_list),
        }
        
    def __call__(self,img, phase):
        return self.data_transforms[phase](img)