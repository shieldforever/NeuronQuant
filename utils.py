import torch
import torch.nn as nn
from modules import *
import numpy as np
from copy import deepcopy
import pickle

class MyShortCut(nn.Module):
    def __init__(self, channels, size, vth) -> None:
        super(MyShortCut, self).__init__()
        self.linears = []
        for i in range(channels):
            self.linears.append(nn.Linear(in_features=size,out_features=size,bias=False))
            nn.init.eye_(self.linears[-1].weight)
            w = self.linears[-1].weight.data.cuda() * vth
            self.linears[-1].weight.data = w
    def forward(self, x):
        channels = len(self.linears)
        output = x
        # consider batchsize
        for i in range(channels):
            output[:,i,:] = self.linears[i](x[:,i,:])
        return output

def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False

def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = IFNeuron(scale=module.v_threshold)
    return model

def merge_conv_bn(conv, bn):
    w = conv.weight.data
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.bias.data
    gamma = bn.weight.data
    
    conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                     conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=True)
    conv.weight.data = w * (gamma / var_sqrt).view(-1, 1, 1, 1)
    conv.bias.data = beta - mean * gamma / var_sqrt
    
    bn.running_mean.zero_()
    bn.running_var.fill_(1)
    bn.bias.data.zero_()
    bn.weight.data.fill_(1)

    return conv, bn

def clear_BNLayer(model):
    conv_name = ''
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = clear_BNLayer(module)
        if module.__class__.__name__ == 'Conv2d':
            conv_name = name
        elif module.__class__.__name__ == 'BatchNorm2d':
            model._modules[conv_name], model._modules[name] = merge_conv_bn(model._modules[conv_name], model._modules[name])
    return model

def delete_useless_layer(layer):
    layer = nn.Sequential(*[module for module in layer if module.__class__.__name__ != 'BatchNorm2d'])
    layer = nn.Sequential(*[module for module in layer if module.__class__.__name__ != 'Dropout'])
    return layer

def mlp_mnist_merge_threshold_with_next_layer(model):
    merge_threshold_with_next_layer(model.relu, model.fc2)

def vgg_merge_threshold_with_next_layer(model):
    first_layer_name = 'layer1'
    first_layer_id = 1
    next_layer_name = 'layer1'
    next_layer_id = 2
    while next_layer_name != 'layer6':
        merge_threshold_with_next_layer(model._modules[first_layer_name][first_layer_id], model._modules[next_layer_name][next_layer_id])
        first_layer_name = next_layer_name
        first_layer_id = next_layer_id + 1
        next_layer_id += 2
        if len(model._modules[next_layer_name]) - 1 <= next_layer_id:
            next_layer_id = 0
            next_layer_name = 'layer' + str(int(next_layer_name[5]) + 1)
    merge_threshold_with_next_layer(model._modules['layer5'][5], model._modules['classifier'][1])
    merge_threshold_with_next_layer(model._modules['classifier'][2], model._modules['classifier'][3])
    merge_threshold_with_next_layer(model._modules['classifier'][4], model._modules['classifier'][5])

def resnet18_merge_threshold_with_next_layer(model):
    featuremap_size = {
        "conv1" : 32,
        "conv2_x" : 32,
        "conv3_x" : 32,
        "conv4_x" : 16,
        "conv5_x" : 8,
    }
    channels = {
        "conv2_x" : 64,
        "conv3_x" : 128,
        "conv4_x" : 256,
        "conv5_x" : 512,
    }
    merge_threshold_with_next_layer(model.conv1[1], model.conv2_x[0].residual_function[0])
    if len(model.conv2_x[0].shortcut._modules) == 0:
        model.conv2_x[0].shortcut = MyShortCut(channels=channels['conv2_x'], size=featuremap_size['conv2_x'], vth=model.conv1[1].v_threshold)
    
    for i in range(2,6):
        cur_layer_name = "conv{}_x".format(i)
        next_layer_name = "conv{}_x".format(i+1)
        merge_threshold_with_next_layer(model._modules[cur_layer_name][0].residual_function[1], model._modules[cur_layer_name][0].residual_function[2]) 
        merge_threshold_with_next_layer(model._modules[cur_layer_name][0].relu, model._modules[cur_layer_name][1].residual_function[0])
        if len(model._modules[cur_layer_name][1].shortcut._modules) == 0:
            model._modules[cur_layer_name][1].shortcut = MyShortCut(channels=model._modules[cur_layer_name][1].residual_function[2].out_channels, size=featuremap_size["conv{}_x".format(i)]//(2 if i > 2 else 1), vth=model._modules[cur_layer_name][0].relu.v_threshold)
        
        merge_threshold_with_next_layer(model._modules[cur_layer_name][1].residual_function[1], model._modules[cur_layer_name][1].residual_function[2]) 
        
        if i < 5:
            merge_threshold_with_next_layer(model._modules[cur_layer_name][1].relu, model._modules[next_layer_name][0].residual_function[0])
            if len(model._modules[next_layer_name][0].shortcut._modules) == 0:
                model._modules[next_layer_name][0].shortcut = MyShortCut(channels=model._modules[next_layer_name][0].residual_function[2].out_channels, size=featuremap_size["conv{}_x".format(i+1)], vth=model._modules[cur_layer_name][1].relu.v_threshold)
    
def merge_threshold_with_next_layer(neuron, conv):
    w = conv.weight.data.cuda()
    w *= neuron.v_threshold
    conv.weight.data = w
    neuron.is_merged = True


def quant_fc_in_once(fc, neuron, bitwidth):
    global exp_device
    w = fc.weight.data # [out_features, in_features]
    if len(fc.state_dict()) == 2:
        w, fc.bias.data, v_threshold = float_to_int(w, fc.bias.data, neuron.v_threshold, bitwidth)
    else:
        w, _, v_threshold = float_to_int(w, neuron.v_threshold, neuron.v_threshold, bitwidth)
    fc.weight.data = w
    return fc, IFNeuron(v_threshold, True, neuron.v_threshold, "fc", exp_device)

def quant_conv2d_in_once(conv, neuron, bitwidth):
    global exp_device
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    w, conv.bias.data, v_threshold = float_to_int(w, conv.bias.data, neuron.v_threshold, bitwidth)
    conv.weight.data = w
    return conv, IFNeuron(v_threshold, True, neuron.v_threshold, "conv", exp_device)

def quant_res_conv_in_once(conv, shortcut, neuron, bitwidth):
    global exp_device
    w_shortcut = shortcut.weight.data.cpu()
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale = (2 ** (bitwidth - 1) - 1) / max(abs(min(torch.min(w), torch.min(w_shortcut))), abs(max(torch.max(w), torch.max(w_shortcut))))
    w, conv.bias.data, v_threshold = multiple_scale(w, conv.bias.data, neuron.v_threshold, scale)
    w_shortcut, shortcut.bias.data, _ = multiple_scale(w_shortcut, shortcut.bias.data, neuron.v_threshold, scale)
    conv.weight.data = w
    shortcut.weight.data = w_shortcut
    return conv, shortcut, IFNeuron(v_threshold, True, neuron.v_threshold, "conv", exp_device)

def quant_res_linear_in_once(conv, shortcut, neuron, bitwidth):
    global exp_device
    w_shortcut = []
    for i in range(conv.out_channels):
        w_shortcut.append(shortcut[i].weight.data)
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale = (2 ** (bitwidth - 1) - 1) / max(abs(torch.min(w)), abs(torch.max(w)))
    
    w, conv.bias.data, v_threshold = multiple_scale(w, conv.bias.data, neuron.v_threshold, scale)
    for i in range(conv.out_channels):
            w_shortcut[i], _, _ = multiple_scale(w_shortcut[i], neuron.v_threshold, neuron.v_threshold, scale)
    conv.weight.data = w
    for i in range(conv.out_channels):
        shortcut[i].weight.data = w_shortcut[i].to(exp_device)
    return conv, shortcut, IFNeuron(v_threshold, True, neuron.v_threshold, "conv", exp_device)

def set_exp_device(device):
    global exp_device
    exp_device = device

score_list = []
score_list_idx = 0
exp_device = "cuda:0"

def cal_score(matrix=1, bias=torch.zeros(1), v_threshold=1, scale=1):
    new_v_threshold = torch.floor(v_threshold * scale)
    new_bias = torch.round(bias * (new_v_threshold / v_threshold))
    new_matrix = torch.round(matrix * (new_v_threshold / v_threshold))
    score = torch.sum(torch.abs(new_matrix/new_v_threshold-matrix/v_threshold)) + torch.abs(new_bias/new_v_threshold-bias/v_threshold) # todo
    return score.item()

def get_score_fc_per_neuron(fc, neuron, bitwidth):
    global score_list, score_list_idx
    w = fc.weight.data.cpu() # [out_features, in_features]
    for i in range(len(w)):
        val = 2**(bitwidth-1)-1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))
        score_list.append([score_list_idx, cal_score(matrix=w[i], v_threshold=neuron.v_threshold, scale=scale)])
        score_list_idx += 1

def get_score_conv2d_per_neuron(conv, neuron, bitwidth):
    global score_list, score_list_idx
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    for i in range(len(w)):
        val = 2**(bitwidth-1)-1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))
        score_list.append([score_list_idx, cal_score(matrix=w[i], bias=conv.bias.data[i], v_threshold=neuron.v_threshold, scale=scale)])
        score_list_idx += 1

def get_score_res_conv_per_neuron(conv, shortcut, neuron, bitwidth):
    global score_list, score_list_idx
    w_shortcut = shortcut.weight.data.cpu()
    w = conv.weight.data.cpu()
    for i in range(len(w)):
        val = 2**(bitwidth-1)-1
        scale = val / max(abs(min(torch.min(w[i]), torch.min(w_shortcut[i]))), abs(max(torch.max(w[i]), torch.max(w_shortcut[i]))))
        score = cal_score(matrix=w[i], bias=conv.bias.data[i], v_threshold=neuron.v_threshold, scale=scale)
        score += cal_score(matrix=w_shortcut[i], bias=shortcut.bias.data[i], v_threshold=neuron.v_threshold, scale=scale)
        score_list.append([score_list_idx, score])
        score_list_idx += 1

def get_score_res_linear_per_neuron(conv, shortcut, neuron, bitwidth):
    global score_list, score_list_idx
    w_shortcut = []
    for i in range(conv.out_channels):
        w_shortcut.append(shortcut[i].weight.data.cpu())
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale_factors = torch.zeros(w.shape[0])
    for i in range(len(w)):
        val = 2**(bitwidth-1)-1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))
        score = cal_score(matrix=w[i], bias=conv.bias.data[i], v_threshold=neuron.v_threshold, scale=scale)
        score += cal_score(matrix=w_shortcut[i], v_threshold=neuron.v_threshold, scale=scale)
        score_list.append([score_list_idx, score])
        score_list_idx += 1


def new_quant_fc_per_neuron(fc, neuron, quant_method):
    global score_list, score_list_idx, exp_device
    w = fc.weight.data.cpu() # [out_features, in_features]
    scale_factors = torch.zeros(w.shape[0])
    for i in range(len(w)):
        val = 2**(score_list[score_list_idx][2]-1)-1
        score_list_idx += 1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))
        if quant_method == "score":
            if len(fc.state_dict()) == 2:
                w[i], fc.bias.data[i], v_threshold = new_multiple_scale(w[i], fc.bias.data[i], neuron.v_threshold, scale)
            else:
                w[i], _, v_threshold = new_multiple_scale(w[i], neuron.v_threshold, neuron.v_threshold, scale)
        else:
            if len(fc.state_dict()) == 2:
                w[i], fc.bias.data[i], v_threshold = multiple_scale(w[i], fc.bias.data[i], neuron.v_threshold, scale)
            else:
                w[i], _, v_threshold = multiple_scale(w[i], neuron.v_threshold, neuron.v_threshold, scale)
        scale_factors[i] = v_threshold
    fc.weight.data = w
    return fc, IFNeuron(scale_factors, True, neuron.v_threshold, "fc", exp_device)

def new_quant_conv2d_per_neuron(conv, neuron, quant_method):
    # conv weight and next neuron to quant
    global score_list, score_list_idx, exp_device
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale_factors = torch.zeros(w.shape[0])
    for i in range(len(w)):
        val = 2**(score_list[score_list_idx][2]-1)-1
        score_list_idx += 1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))
        if quant_method == "score":
            w[i], conv.bias.data[i], v_threshold = new_multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale)
        else:
            w[i], conv.bias.data[i], v_threshold = multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale) # ???? neuron.v_threshold不加索引吗[i]？
        scale_factors[i] = v_threshold
    conv.weight.data = w
    return conv, IFNeuron(scale_factors, True, neuron.v_threshold, "conv", exp_device)

def new_quant_res_conv_per_neuron(conv, shortcut, neuron, quant_method):
    global score_list, score_list_idx, exp_device
    w_shortcut = shortcut.weight.data.cpu()
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale_factors = torch.zeros(w.shape[0])
    for i in range(len(w)):
        val = 2**(score_list[score_list_idx][2]-1)-1
        score_list_idx += 1
        scale = val / max(abs(min(torch.min(w[i]), torch.min(w_shortcut[i]))), abs(max(torch.max(w[i]), torch.max(w_shortcut[i]))))

        if quant_method == "score":
            w[i], conv.bias.data[i], v_threshold = new_multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale)
            w_shortcut[i], shortcut.bias.data[i], _ = new_multiple_scale(w_shortcut[i], shortcut.bias.data[i], neuron.v_threshold, scale)
        else:
            w[i], conv.bias.data[i], v_threshold = multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale)
            w_shortcut[i], shortcut.bias.data[i], _ = multiple_scale(w_shortcut[i], shortcut.bias.data[i], neuron.v_threshold, scale)
        scale_factors[i] = v_threshold
    conv.weight.data = w
    shortcut.weight.data = w_shortcut
    return conv, shortcut, IFNeuron(scale_factors, True, neuron.v_threshold, "conv", exp_device)

def new_quant_res_linear_per_neuron(conv, shortcut, neuron, quant_method):
    #shortcut represent liner
    global score_list, score_list_idx, exp_device
    w_shortcut = []
    for i in range(conv.out_channels):
        w_shortcut.append(shortcut[i].weight.data)
    w = conv.weight.data.cpu() # [out_channels, in_channels, kernel_size, kernel_size]
    scale_factors = torch.zeros(w.shape[0])
    for i in range(len(w)):
        val = 2**(score_list[score_list_idx][2]-1)-1
        score_list_idx += 1
        scale = val / max(abs(torch.min(w[i])), abs(torch.max(w[i])))

        if quant_method == "score":
            w[i], conv.bias.data[i], v_threshold = new_multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale)
            w_shortcut[i], _, _ = new_multiple_scale(w_shortcut[i], neuron.v_threshold, neuron.v_threshold, scale)
        else:
            w[i], conv.bias.data[i], v_threshold = multiple_scale(w[i], conv.bias.data[i], neuron.v_threshold, scale)
            w_shortcut[i], _, _ = multiple_scale(w_shortcut[i], neuron.v_threshold, neuron.v_threshold, scale)
        scale_factors[i] = v_threshold
    conv.weight.data = w
    for i in range(conv.out_channels):
        shortcut[i].weight.data = w_shortcut[i].to(exp_device)
    return conv, shortcut, IFNeuron(scale_factors, True, neuron.v_threshold, "conv", exp_device)


def new_multiple_scale(matrix, bias, v_threshold, scale):
    new_v_threshold = torch.floor(v_threshold * scale) 
    new_bias = torch.round(bias * (new_v_threshold / v_threshold))
    new_matrix = torch.round(matrix * (new_v_threshold / v_threshold))
    return new_matrix, new_bias, new_v_threshold

def multiple_scale(matrix, bias, v_threshold, scale):
    return torch.round(matrix * scale), torch.round(bias * scale), torch.floor(v_threshold * scale) 

min_loss_vals = []

def float_to_int(matrix, bias, v_threshold, bitwidth):
    min_val = torch.min(matrix)
    max_val = torch.max(matrix)
    max_val = max(abs(min_val), abs(max_val))
    scale = (2 ** (bitwidth - 1) - 1) / max_val
    return multiple_scale(matrix, bias, v_threshold, scale)


def get_bitwidth(matrix, thresholds):
    # get bitwidth of w
    sum = 0
    for node in range(len(matrix)):
        bitwidth = len(bin(abs(int(thresholds[node])))) - 2
        for w in matrix[node]:
            bitwidth = max(bitwidth, len(bin(abs(int(w)))) - 2)
        sum += bitwidth
    return sum


def quant_by_loss(matrix, thresholds):
    def getloss(W, t, p):
        sum = abs(t / p - np.round(t / p))
        for w in W:
            sum += abs(w / p - np.round(w / p))
        return sum

    for node in range(len(matrix)):
        loss_data = []
        min_loss_index = 32
        for i in range(len(matrix[node])):
            matrix[node][i] = np.round(matrix[node][i] / min_loss_index)
        thresholds[node] = np.round(thresholds[node] / min_loss_index)
    return matrix, thresholds


def replace_activation_by_MPLayer(model, presim_len, sim_len):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_MPLayer(module, presim_len, sim_len)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = MPLayer(v_threshold=module.up.item(), presim_len=presim_len, sim_len=sim_len)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=8., t=t)
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def error(info):
    print(info)
    exit(1)
    
