from utils import *
import utils
from NetworkFunction import *
import argparse
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_ImageNet, PreProcess_MNIST
from Models.ResNet import *
from Models.VGG import *
from Models.MLP_MNIST import *
import torch
import random
import os, time
import numpy as np

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='./dataset', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='./pths/', help='Directory where the model is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=300, help='Training Epochs of ANNs')
    parser.add_argument('--activation_floor', type=str, default='QCFS', help='ANN activation modules')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=512, help='Batch size')
    parser.add_argument('--L', type=int, default=4, help='Quantization level of QCFS')
    parser.add_argument('--sim_len', type=int, default=32, help='Simulation length of SNNs')
    parser.add_argument('--presim_len', type=int, default=0, help='Pre Simulation length of SRP')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/datasets/cluster/public/ImageNet/ILSVRC2012_train', help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/datasets/cluster/public/ImageNet/ILSVRC2012_val', help='Directory where the ImageNet test dataset is saved')    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits for layer-wise')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1,2,3')
    parser.add_argument('--quant_method', type=str, default="origin") # origin/score
    parser.add_argument('--strategy', type=str, default="random") # random/score
    parser.add_argument('--whole_layer_quant', default=False, action='store_true')
    parser.add_argument('--bits_rate', type=float, default=1.0, help='Quantization bits rate')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    set_exp_device(args.device)
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    
    cls = 100
    cap_dataset = 10000
    
    if args.dataset == 'CIFAR10':
        cls = 10
    elif args.dataset == 'CIFAR100':
        cls = 100
    elif args.dataset == 'ImageNet':
        cls = 1000
        cap_dataset = 50000
    elif args.dataset == "MNIST":
        cls = 10
    
    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls, bias=False)
    elif args.net_arch == 'vggn':
        model = vgg16_normed(num_classes=cls, bias=False)
    elif args.net_arch == 'MLP_MNIST':
        model = mlp_mnist()
    else:
        error('unable to find model ' + args.net_arch)
    
    
    if args.activation_floor == 'QCFS':
        model = replace_activation_by_floor(model, args.L)
    else:
        error('unable to find activation floor: ' + args.activation_floor)
    
    if args.dataset == 'CIFAR10':
        train, test = PreProcess_Cifar10(args.datadir, args.batchsize)
    elif args.dataset == 'CIFAR100':
        train, test = PreProcess_Cifar100(args.datadir, args.batchsize)
    elif args.dataset == 'ImageNet':
        train, test = PreProcess_ImageNet(args.datadir, args.batchsize, train_dir=args.train_dir, test_dir=args.test_dir)
    elif args.dataset == 'MNIST':
        train, test = PreProcess_MNIST(args.datadir, args.batchsize)
    else:
        error('unable to find dataset ' + args.dataset)


    if args.load_model_name != 'None':
        print(f'=== Load Pretrained ANNs ===')
        model.load_state_dict(torch.load(os.path.join("pths", args.load_model_name + '.pth')))  
    if args.direct_training is True:
        print(f'=== Start Training ANNs ===')
        save_name = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_L' + str(args.L) + '.pth'
        model = train_ann(train, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device, save_name=save_name)
        torch.save(model.state_dict(), os.path.join("pths", args.net_arch + '_'+ args.dataset + '.pth'))
    
    clear_BNLayer(model)
    # Delete useless layers like BN, Dropout.
    for name, module in model._modules.items():
        if args.net_arch == "resnet18":
            if module.__class__.__name__ == "AdaptiveMaxPool2d":
                break
            if name == "conv1":
                model._modules[name] = delete_useless_layer(model._modules[name])
            else:
                for subname, submodule in module._modules.items():
                    model._modules[name]._modules[subname].residual_function = delete_useless_layer(model._modules[name]._modules[subname].residual_function)
                    model._modules[name]._modules[subname].shortcut = delete_useless_layer(model._modules[name]._modules[subname].shortcut)

        elif args.net_arch == "vgg16":
            model._modules[name] = delete_useless_layer(model._modules[name])
        else:
            pass
    

    replace_activation_by_MPLayer(model,presim_len=args.presim_len,sim_len=args.sim_len)

    replace_MPLayer_by_neuron(model)
    
    if args.net_arch == 'vgg16':
        vgg_merge_threshold_with_next_layer(model)
    elif args.net_arch == "resnet18":
        resnet18_merge_threshold_with_next_layer(model)
    elif args.net_arch == "MLP_MNIST":
        mlp_mnist_merge_threshold_with_next_layer(model)
        
    

    print(f'=== Start Calc Scores ===')
    if args.net_arch == "vgg16":
        for name, module in model._modules.items():
            if 'layer' in name:
                for i in range(0, len(model._modules[name]) - 1, 2):
                    get_score_conv2d_per_neuron(model._modules[name][i], model._modules[name][i + 1], 4)
            else:
                for i in range(1, len(model._modules[name]) - 1, 2):
                    get_score_fc_per_neuron(model._modules[name][i], model._modules[name][i + 1], 4)
            print(f'{name} calc finish')
    elif args.net_arch == "resnet18":
        for name, module in model._modules.items():
            if name == "conv1":
                get_score_conv2d_per_neuron(model._modules[name][0], model._modules[name][1], 4)
            elif "conv" in name:
                for j in range(2):
                    get_score_conv2d_per_neuron(model._modules[name][j].residual_function[0], model._modules[name][j].residual_function[1], 4)
                    if model._modules[name][j].shortcut.__class__.__name__ == "MyShortCut":
                        get_score_res_linear_per_neuron(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut.linears, model._modules[name][j].relu, 4)
                    else:
                        get_score_res_conv_per_neuron(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut[0], model._modules[name][j].relu, 4)
            else:
                pass
            print(f'{name} calc finish')
    elif args.net_arch == "MLP_MNIST":
        get_score_fc_per_neuron(model.fc1, model.relu, 4)
        print(f'{name} calc finish')
    rate_list = [1-args.bits_rate, args.bits_rate]
    bitwidth_list = [4,8] 
    length = len(score_list)
    range_list = []
    range_list.append([0, int(length*rate_list[0])])
    for i in range(1, len(rate_list)):
        range_list.append([range_list[i-1][1], range_list[i-1][1]+int(length*rate_list[i])])
    range_list[-1][1] = length
    
    strategy = args.strategy
    if strategy == "score":
        score_list = sorted(score_list, key=lambda x: x[1]) # sort score(True)
        for i in range(len(rate_list)):
            for j in range(range_list[i][0], range_list[i][1]):
                score_list[j].append(bitwidth_list[i])
        score_list = sorted(score_list, key=lambda x: x[0]) # sort score(True)
    elif strategy == "random":
        random.shuffle(score_list)
        for i in range(len(rate_list)):
            for j in range(range_list[i][0], range_list[i][1]):
                score_list[j].append(bitwidth_list[i])
        score_list = sorted(score_list, key=lambda x: x[0]) # sort score(True)
    else:
        pass
    utils.score_list_idx = 0

    print(f'=== Start Quanting ===')
    if args.net_arch == "vgg16":
        layer_idx = 1
        for name, module in model._modules.items():
            start_time = time.time()   
            if 'layer' in name:
                for i in range(0, len(model._modules[name]) - 1, 2):
                    if args.whole_layer_quant:
                        model._modules[name][i], model._modules[name][i + 1] = quant_conv2d_in_once(model._modules[name][i], model._modules[name][i + 1], args.bits)
                    else:
                        model._modules[name][i], model._modules[name][i + 1] = new_quant_conv2d_per_neuron(model._modules[name][i], model._modules[name][i + 1], args.quant_method)
            else:
                for i in range(1, len(model._modules[name]) - 1, 2):
                    if args.whole_layer_quant:
                        model._modules[name][i], model._modules[name][i + 1] = quant_fc_in_once(model._modules[name][i], model._modules[name][i + 1], args.bits)
                    else:
                        model._modules[name][i], model._modules[name][i + 1] = new_quant_fc_per_neuron(model._modules[name][i], model._modules[name][i + 1], args.quant_method)
                       
            end_time = time.time()
            print(f'Quanting {name} cost {end_time - start_time} seconds')
    
    elif args.net_arch == "resnet18":
        for name, module in model._modules.items():
            start_time = time.time()
            if name == "conv1":
                if args.whole_layer_quant:
                    model._modules[name][0], model._modules[name][1] = quant_conv2d_in_once(model._modules[name][0], model._modules[name][1], args.bits)
                else:
                    model._modules[name][0], model._modules[name][1] = new_quant_conv2d_per_neuron(model._modules[name][0], model._modules[name][1], args.quant_method)
            elif "conv" in name:
                for j in range(2):
                    if args.whole_layer_quant:
                        model._modules[name][j].residual_function[0], model._modules[name][j].residual_function[1] = quant_conv2d_in_once(model._modules[name][j].residual_function[0], model._modules[name][j].residual_function[1], args.bits)
                    else:
                        model._modules[name][j].residual_function[0], model._modules[name][j].residual_function[1] = new_quant_conv2d_per_neuron(model._modules[name][j].residual_function[0], model._modules[name][j].residual_function[1], args.quant_method)
                    
                    if model._modules[name][j].shortcut.__class__.__name__ == "MyShortCut":
                        if args.whole_layer_quant:
                            model._modules[name][j].residual_function[2], model._modules[name][j].shortcut.linears, model._modules[name][j].relu = quant_res_linear_in_once(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut.linears, model._modules[name][j].relu, args.bits)
                        else:
                            model._modules[name][j].residual_function[2], model._modules[name][j].shortcut.linears, model._modules[name][j].relu = new_quant_res_linear_per_neuron(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut.linears, model._modules[name][j].relu, args.quant_method)
                    else:
                        if args.whole_layer_quant:
                            model._modules[name][j].residual_function[2], model._modules[name][j].shortcut[0], model._modules[name][j].relu = quant_res_conv_in_once(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut[0], model._modules[name][j].relu, args.bits)
                        else:
                            model._modules[name][j].residual_function[2], model._modules[name][j].shortcut[0], model._modules[name][j].relu = new_quant_res_conv_per_neuron(model._modules[name][j].residual_function[2], model._modules[name][j].shortcut[0], model._modules[name][j].relu, args.quant_method)
                    
            else:
                pass
            end_time = time.time()
            print(f'Quanting {name} cost {end_time - start_time} seconds')
    elif args.net_arch == "MLP_MNIST":
        if args.whole_layer_quant:
            model.fc1, model.relu = quant_fc_in_once(model.fc1, model.relu, args.bits)
        else:
            model.fc1, model.relu = new_quant_fc_per_neuron(model.fc1, model.relu, args.quant_method)
    
    torch.cuda.empty_cache()


    new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device)

    t = 1
    while t < args.sim_len:
        print(f'time step {t}, Accuracy = {(new_acc[t-1] / cap_dataset):.4f}')
        t *= 2
    print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len-1] / cap_dataset):.4f}')

