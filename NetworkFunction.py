from torch import nn
import torch
from tqdm import tqdm
from utils import *
import pickle
from spikingjelly.clock_driven import encoding

def mp_test(test_dataloader, model, net_arch, presim_len, sim_len, device):
    new_tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            new_spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            
            for t in range(presim_len+sim_len):
                out = model(img)
                
                if t >= presim_len:
                    new_spikes += out
                    new_tot[t-presim_len] += (label==new_spikes.max(1)[1]).sum().item()
                   
    return new_tot


def train_ann(train_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name):
    encoder = encoding.PoissonEncoder()
    model = model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0
        model.train()
        for img, label in train_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            if model.__class__.__name__ == "MLP_MNIST":
                out = model(encoder(img).float())
            else:
                out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            lenth += len(img)
        acc = eval_ann(test_dataloader, model, device)
        print(f"ANNs training Epoch {epoch}: Val_loss: {epoch_loss/lenth} Acc: {acc}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_name)
        
        scheduler.step()  
          
    return model

def eval_ann(test_dataloader, model, device):
    encoder = encoding.PoissonEncoder()
    tot = 0
    model.eval()
    model.cuda(device)
    
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            if model.__class__.__name__ == "MLP_MNIST":
                out = model(encoder(img).float())
            else:
                out = model(img)
            tot += (label==out.max(1)[1]).sum().item()
           
    return tot

def eval_snn(test_dataloader, model, sim_len, device):
    #evaluate acc of every t step
    encoder = encoding.PoissonEncoder()
    tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_dataloader):
            spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            for t in range(sim_len):
                if model.__class__.__name__ == "MLP_MNIST":
                    out = model(encoder(img).float())
                else:
                    out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum().item()
            reset_net(model)   
            print("{} finished".format(round((idx+1)/len(test_dataloader), 2)))

    return tot
