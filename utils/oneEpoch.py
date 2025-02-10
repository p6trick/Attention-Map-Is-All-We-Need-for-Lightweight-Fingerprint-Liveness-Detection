import torch
from keras.utils import to_categorical

def train_one(x,y, model, criterion, optimizer, ACE, loss_coef=[1,0.2]):
    correct = 0
    att_correct = 0
    criterion = criterion.cuda()
    outputs,_,_,att_outputs = model(x)
    
    loss1 = criterion(outputs, y)
    loss2 = criterion(att_outputs, y)
    loss = loss_coef[0]*loss1 + loss2*loss_coef[1]
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    _,pred = torch.max(outputs.data,1)
    _,att_pred = torch.max(att_outputs.data, 1)
    correct += (pred == y).sum().item()
    att_correct += (att_pred == y).sum().item()
    total = len(y)
    
    return correct, loss.item(), total, ace[1],ace[3], att_correct

def val_one(x,y, model, criterion, ACE):
    correct = 0
    att_correct = 0
    outputs,_,_, att_outputs = model(x)
   
    loss1 = criterion(outputs, y)
    loss2 = criterion(att_outputs, y)
    loss = loss1 + loss2*0.2
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    total = len(y)
    _, pred = torch.max(outputs.data,1)
    _,att_pred = torch.max(att_outputs.data, 1)

    correct += (pred == y).sum().item()
    att_correct += (att_pred == y).sum().item()

    return correct, loss.item(), total, ace[1], ace[3], att_correct

def val_one_new(x,y, model, criterion, ACC, ACE):
    correct = 0
    att_correct = 0
    outputs = model(x)
   
    # loss1 = criterion(outputs, y)
    # loss2 = criterion(att_outputs, y)
    # loss = loss1 + loss2*0.2
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    # l = to_categorical(y.detach().cpu().numpy())
    # p = to_categorical(outputs.argmax(dim=1).detach().cpu())
    
    ace = ACE(outputs.argmax(dim=1).detach().cpu(), y.detach().cpu(), labels=[True, False])
    acc = ACC(outputs.argmax(dim=1).detach().cpu(), y.detach().cpu())
    
    total = len(y)
    _, pred = torch.max(outputs.data,1)
    # _,att_pred = torch.max(att_outputs.data, 1)

    correct += (pred == y).sum().item()
    # att_correct += (att_pred == y).sum().item()

    return correct, acc, total, ace #, att_correct

def train_compare(x,y, model, criterion, optimizer, ACE, loss_coef=[1,0.2]):
    correct = 0
    
    criterion = criterion.cuda()
    outputs = model(x)
    
    loss = criterion(outputs, y)
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _,pred = torch.max(outputs.data,1)
    correct += (pred == y).sum().item()
    total = len(y)
    
    return correct, loss.item(), total, ace[1], ace[3]

def val_compare(x,y, model, criterion, ACE):
    correct = 0
    
    outputs = model(x)
   
    loss = criterion(outputs, y)
    
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    total = len(y)
    _, pred = torch.max(outputs.data,1)

    correct += (pred == y).sum().item()
    return correct, loss.item(), total, ace[1], ace[3]

def ablation_train_one(x,y, model, criterion, optimizer, ACE):
    correct = 0
    
    criterion = criterion.cuda()
    outputs = model(x)
    
    loss = criterion(outputs, y)
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    _,pred = torch.max(outputs.data,1)
    correct += (pred == y).sum().item()
    total = len(y)
    
    return correct, loss.item(), total, ace[1],ace[3]

def ablation_val_one(x,y, model, criterion, ACE):
    correct = 0
    outputs = model(x)
   
    loss = criterion(outputs, y)
    
    if len(y.shape) == 2:
        y = y.argmax(dim=1)
    
    ace = ACE(outputs.argmax(dim=1), y)
    
    total = len(y)
    _, pred = torch.max(outputs.data,1)
    correct += (pred == y).sum().item()

    return correct, loss.item(), total, ace[1], ace[3]