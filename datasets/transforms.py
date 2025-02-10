import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

class FiveCrop:
    def __init__(self, sensor, crop_size=224):
        

        if sensor in ['CrossMatch']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((375, 400)),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        elif sensor in ['HiScan']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(500),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        else:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
            

    def transforms(self, Data):
        new_x_train = []
        new_x_test = []
        print('...Cropping x_train...')
        for idx,i in enumerate(tqdm(Data['x_train'])):
            i = self.fivecrop(np.abs((255-i)/255.).astype(np.float32))
            new_x_train.append(i)
            
        print('...Cropping x_test...')
        for idx,i in enumerate(tqdm(Data['x_test'])):
            i = self.fivecrop_test(np.abs((255-i)/255.).astype(np.float32))
            new_x_test.append(i)
        Data['x_train'] = new_x_train
        Data['x_test'] = new_x_test
        
        print(f"x_train.shape: {Data['x_train'][0].shape}")
        print(f"x_test.shape: {Data['x_test'][0].shape}")
        
        return Data
    
    def white_transforms(self, Data):
        new_x_train = []
        new_x_test = []
        print('...Cropping x_train...')
        for idx,i in enumerate(tqdm(Data['x_train'])):
            i = self.fivecrop((i/255.).astype(np.float32))
            new_x_train.append(i)
            
        print('...Cropping x_test...')
        for idx,i in enumerate(tqdm(Data['x_test'])):
            i = self.fivecrop_test((i/255.).astype(np.float32))
            new_x_test.append(i)
        Data['x_train'] = new_x_train
        Data['x_test'] = new_x_test
        
        print(f"x_train.shape: {Data['x_train'][0].shape}")
        print(f"x_test.shape: {Data['x_test'][0].shape}")
        
        return Data
    
    
class WhiteFiveCrop:
    def __init__(self, sensor, crop_size=224):
        

        if sensor in ['CrossMatch']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((375, 400)),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmin(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        elif sensor in ['HiScan']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(500),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmin(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        else:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.FiveCrop(crop_size),
                            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            transforms.Lambda(lambda best: best[torch.argmin(torch.tensor([b.sum() for b in best]))]),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
            

    def transforms(self, Data):
        new_x_train = []
        new_x_test = []
        print('...Cropping x_train...')
        for idx,i in enumerate(tqdm(Data['x_train'])):
            i = self.fivecrop((i/255.).astype(np.float32))
            new_x_train.append(i)
            
        print('...Cropping x_test...')
        for idx,i in enumerate(tqdm(Data['x_test'])):
            i = self.fivecrop_test((i/255.).astype(np.float32))
            new_x_test.append(i)
        Data['x_train'] = new_x_train
        Data['x_test'] = new_x_test
        
        print(f"x_train.shape: {Data['x_train'][0].shape}")
        print(f"x_test.shape: {Data['x_test'][0].shape}")
        
        return Data
    
    
class noFiveCrop:
    def __init__(self, sensor, crop_size=224):
        

        if sensor in ['CrossMatch']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((375, 400)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        elif sensor in ['HiScan']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(500),
                            transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        else:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
            

    def transforms(self, Data):
        new_x_train = []
        new_x_test = []
        print('...Cropping x_train...')
        for idx,i in enumerate(tqdm(Data['x_train'])):
            i = self.fivecrop((i/255.).astype(np.float32))
            new_x_train.append(i)
            
        print('...Cropping x_test...')
        for idx,i in enumerate(tqdm(Data['x_test'])):
            i = self.fivecrop_test((i/255.).astype(np.float32))
            new_x_test.append(i)
        Data['x_train'] = new_x_train
        Data['x_test'] = new_x_test
        
        print(f"x_train.shape: {Data['x_train'][0].shape}")
        print(f"x_test.shape: {Data['x_test'][0].shape}")
        
        return Data


class NoCrop:
    def __init__(self, sensor, crop_size=224):
        

        if sensor in ['CrossMatch']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.Resize((224,224)),
                            # transforms.RandomCrop(crop_size),
                            # transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            # transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                 transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        elif sensor in ['HiScan']:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.Resize((224,224)),
                            # transforms.RandomCrop(crop_size),
                            # transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            # transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                 transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
        else:
            self.fivecrop = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.Resize((224,224)),
                            # transforms.RandomCrop(crop_size),
                            # transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
                            # transforms.Lambda(lambda best: best[torch.argmax(torch.tensor([b.sum() for b in best]))]),
                 transforms.ToTensor(),
                            transforms.Normalize([0.8],[0.3])
                        ])
            self.fivecrop_test = self.fivecrop
            

    def transforms(self, Data):
        new_x_train = []
        new_x_test = []
        print('...Cropping x_train...')
        for idx,i in enumerate(tqdm(Data['x_train'])):
            i = self.fivecrop(np.abs((255-i)/255.).astype(np.float32))
            new_x_train.append(i)
            
        print('...Cropping x_test...')
        for idx,i in enumerate(tqdm(Data['x_test'])):
            i = self.fivecrop_test(np.abs((255-i)/255.).astype(np.float32))
            new_x_test.append(i)
        Data['x_train'] = new_x_train
        Data['x_test'] = new_x_test
        
        print(f"x_train.shape: {Data['x_train'][0].shape}")
        print(f"x_test.shape: {Data['x_test'][0].shape}")
        
        return Data
    