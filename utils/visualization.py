import matplotlib.pyplot as plt
import torch
import random
    
def plot_cropped(train_img, test_img):
    
    idx = random.randrange(len(train_img))
    plt.figure(figsize=(6,4))
    plt.subplot(1, 2, 1)
    plt.imshow(torch.tensor(train_img[idx]).permute(1,2,0).clone().detach().numpy(), cmap='gray')
    plt.title('train img')
    plt.axis('off')
    
    idx = random.randrange(len(test_img))
    plt.subplot(1, 2, 2)
    plt.imshow(torch.tensor(test_img[idx]).permute(1,2,0).clone().detach().numpy(), cmap='gray')
    plt.title('test img')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    
    
def plot_history(history):
    x = range(len(history['train']['acc']))
    plt.figure(figsize=(15,4))
    plt.subplot(1, 3, 1)
    plt.plot(x, history['train']['acc'], '--', label = 'train_acc') 
    plt.plot(x, history['val']['acc'], 'r^-', label = 'eval_acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 3, 2)
    plt.plot(x, history['train']['ace'], '--', label = 'train_ace') 
    plt.plot(x, history['val']['ace'], 'r^-', label = 'eval_ace')
    plt.legend()
    plt.title('ACE')
    
    plt.subplot(1, 3, 3)
    plt.plot(x, history['train']['ace'], 'g', label = 'learning_rate') 
    plt.title('learning_rate')
    
    plt.legend() # 범례 표시
    plt.show()
    


def plot_result(t_img, attimg, att):
    plt.subplot(1, 3, 1)                
    plt.imshow(torch.tensor(t_img).clone().detach().squeeze(0).permute(1,2,0).cpu().numpy(), cmap='gray')
    plt.title('input img')
    plt.tight_layout()
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(torch.tensor(t_img).clone().detach().squeeze(0).permute(1,2,0).cpu().numpy(),alpha=0.8, cmap='gray')
    plt.imshow(att.clone().detach().reshape(1,224,224).permute(1,2,0).cpu().numpy(),alpha=0.6)
    plt.title('att + img')
    plt.tight_layout()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(att.clone().detach().reshape(1,224,224).permute(1,2,0).cpu().numpy())
    plt.title('att')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def plot_att(config, img, state):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    config.model.load_state_dict(state)
    config.model.eval()
    with torch.no_grad():
        img = img.reshape(-1,1,224,224).to(config.device)
        start.record()
        
        output, attimg, att, att_out = config.model(img)
        
        end.record()
        torch.cuda.synchronize()
        
        print(f'inference time: {start.elapsed_time(end):.3f}')
        print(f'output: {output}')
    plot_result(img, attimg, att)

    return img, attimg, att
    
    
