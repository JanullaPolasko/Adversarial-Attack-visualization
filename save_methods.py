import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from datapath import my_path
from proximity_methods import  compute_method_ratio, compute_method_projection
from proximity_utils import get_layers
import numpy as np
from visualization_utils import plot_with_dev_subplot, plot_with_dev, plot_projection
from adversarial_utils import  load_model_eval
from datasets import load_data


def configuration(network_type, dataset):
    configurations = {
    ('FC', 'MNIST'): {
        'targets': [(4, 9), (1, 8), (3, 8)],
        'k': 300 },
    ('CONV', 'MNIST'): {
        'targets': [(4, 9), (1, 4), (2, 8)],
        'k': 300 },
    ('CONV', 'CIFAR10'): {
        'targets': [(1, 9), (8, 0), (7, 5)],
        'k': 300 },
    ('CONV', 'SVHN'): {
        'targets': [(2, 3), (1, 4), (3, 5)],
        'k': 300 },
    ('CONV', 'FMNIST'): {
        'targets': [(7, 5), (9, 7), (2, 6)],
        'k': 300 },
    ('RESNET', 'CIFAR10'): {
        'targets': [(8, 0), (3, 5), (9, 1)],
        'k': 100 },
    ('RESNET', 'MNIST'): {
        'targets': [(1, 4), (4, 8), (5, 3)],
        'k': 100 },
    ('VIT', 'MNIST'): {
        'targets': [],
        'k': 100 },
    ('VIT', 'CIFAR10'): {
        'targets': [],
        'k': 100 }, }
    
    config = configurations.get((network_type, dataset), None)
 
    _, _, _, classes = load_data(dataset)
    _,  model = load_model_eval(dataset, network_type, run = 0)
    _, _, x_ticks,_ = get_layers(model)

    targets = config['targets']
    k = config['k'] 
    leg_projection = []
    leg_ratio = []

    
    for orig_class, pred_class in targets:
        help = []
        help.append(rf'${classes[orig_class]}$ ($C_{{{classes[orig_class]}}} \rightarrow C_{{{classes[pred_class]}}}$)')
        help.append(rf'${classes[pred_class]}$ ($C_{{{classes[orig_class]}}} \rightarrow C_{{{classes[pred_class]}}}$)')
        leg_projection.append(help)
        leg_ratio.append(rf'$C_{{{classes[orig_class]}}} \rightarrow C_{{{classes[pred_class]}}}$')

    
    return targets, x_ticks, leg_ratio, leg_projection, k



def using_projection(network_type, dataset, attack, run = 0, plot= True):
    
    targets, x_ticks, _, leg,  k = configuration(network_type, dataset)

    for i in range(len(targets)):
        data_avg, data_dev = [], []
        projection_method = compute_method_projection(dataset, network_type, attack, orig_class=targets[i][0], pred_class=targets[i][1], k=k)
        data_avg.append(projection_method['orig_avg'])
        data_avg.append(projection_method['pred_avg'])
        data_dev.append(projection_method['orig_std'])
        data_dev.append(projection_method['pred_std'])
        projs = projection_method['layer_projections']
        
        
        if plot:
            #PLOT 1
            plot_with_dev(x=[np.arange(len(d))+1 for d in data_avg],
                          y_avg= data_avg, y_dev= data_dev, x_ticks=x_ticks,
                            title=f'{attack} attack, {network_type} model on  {dataset}', legend= [leg[i][0], leg[i][1]],
                            output_path=my_path() + f'/distances/projected/net_{network_type}_{dataset}_attack_{attack}_{targets[i]}', k = 0.5, fig_size=(12,4))
                
                
            # PLOT 2
            #plot projection images
            plot_projection(output_path =my_path() + f'/distances/projected/projection_{network_type}_{dataset}_attack_{attack}_{targets[i]}.png',
                           x_ticks = x_ticks, color_dataset =  dataset in ('CIFAR10', 'SVHN'), projs = projs )

def using_ratio(network_type, dataset, attack, run = 0, plot = True):
    
    targets, x_ticks, leg ,leg2,  k = configuration(network_type, dataset)

    data_avg, data_dev = [], []
    for i in range(len(targets)):
        ratio_method = compute_method_ratio(dataset, network_type, attack, orig_class=targets[i][0], pred_class=targets[i][1], k=k)
        data_avg.append(ratio_method['orig_avg'])
        data_avg.append(ratio_method['pred_avg'])
        data_dev.append(ratio_method['orig_std'])
        data_dev.append(ratio_method['pred_std'])

    if plot:
        #PLOT 1
        plot_with_dev_subplot([np.arange(len(d))+1 for d in data_avg] * len(targets)*2, data_avg, data_dev, x_ticks=x_ticks, title=f'{attack} attack, {network_type} model on  {dataset}', legend=np.ravel(leg2), f_size=(12, 4), 
                                name=my_path() + f'/distances/ratio/net_{network_type}_{dataset}_attack_{attack}_Dev')
        # #PLOT 2
        #staci mi iba ako sa zmensuje pocet spravnych susedov spravnej triedy netreba ajopacne
        plot_with_dev(x=[np.arange(len(d))+1 for d in data_avg[::2]], y_avg=data_avg[::2], y_dev=data_dev[::2], x_ticks=x_ticks, legend=leg, y_lim=[0, k], 
                        x_label='Network Layers', y_label=f'Neighbors for True Class', title=f'{attack} Attack Analysis ({dataset} {network_type})',
                        output_path=my_path() + f'/distances/ratio/net_{network_type}_{dataset}_attack_{attack}')  
            



if __name__ == "__main__":

    # using_ratio( "RESNET", "MNIST" ,"Linf")
    # using_ratio( "RESNET", "MNIST" ,"L0")
    # using_ratio( "RESNET", "MNIST" ,"L1")
    # using_ratio( "RESNET", "MNIST" ,"L2")

    # using_ratio( "RESNET", "CIFAR10" ,"L0")
    # using_ratio( "RESNET", "CIFAR10" ,"L1")
    # using_ratio( "RESNET", "CIFAR10" ,"L2")
    # using_ratio( "RESNET", "CIFAR10" ,"Linf")

    using_ratio( "FC", "MNIST" ,"L0")
    # using_ratio( "FC", "MNIST" ,"L1")
    # using_ratio( "FC", "MNIST" ,"L2")
    # using_ratio( "FC", "MNIST" ,"Linf")

    # using_ratio( "CONV", "MNIST" ,"L0")
    # using_ratio( "CONV", "MNIST" ,"L1")
    # using_ratio( "CONV", "MNIST" ,"L2")
    # using_ratio( "CONV", "MNIST" ,"Linf")

    # using_ratio( "CONV", "CIFAR10" ,"L0")
    # using_ratio( "CONV", "CIFAR10" ,"L1")
    # using_ratio( "CONV", "CIFAR10" ,"L2")
    # using_ratio( "CONV", "CIFAR10" ,"Linf")

    # using_ratio( "CONV", "SVHN" ,"L0")
    # using_ratio( "CONV", "SVHN" ,"L1")
    # using_ratio( "CONV", "SVHN" ,"L2")
    # using_ratio( "CONV", "SVHN" ,"Linf")

    # using_ratio( "CONV", "FMNIST" ,"L0")
    # using_ratio( "CONV", "FMNIST" ,"L1")
    # using_ratio( "CONV", "FMNIST" ,"L2")
    # using_ratio( "CONV", "FMNIST" ,"Linf")

    using_projection( "FC", "MNIST" ,"L0")
    using_projection( "FC", "MNIST" ,"L1")
    using_projection( "FC", "MNIST" ,"L2")
    using_projection( "FC", "MNIST" ,"Linf")
    


    
    using_projection( "CONV", "MNIST" ,"L0")
    using_projection( "CONV", "MNIST" ,"L1")
    using_projection( "CONV", "MNIST" ,"L2")
    using_projection( "CONV", "MNIST" ,"Linf")

    using_projection( "CONV", "CIFAR10" ,"L0")
    using_projection( "CONV", "CIFAR10" ,"L1")
    using_projection( "CONV", "CIFAR10" ,"L2")
    using_projection( "CONV", "CIFAR10" ,"Linf")

    using_projection( "CONV", "SVHN" ,"L0")
    using_projection( "CONV", "SVHN" ,"L1")
    using_projection( "CONV", "SVHN" ,"L2")
    using_projection( "CONV", "SVHN" ,"Linf")

    using_projection( "CONV", "FMNIST" ,"L0")
    using_projection( "CONV", "FMNIST" ,"L1")
    using_projection( "CONV", "FMNIST" ,"L2")
    using_projection( "CONV", "FMNIST" ,"Linf")


    using_projection( "RESNET", "MNIST" ,"Linf")
    using_projection( "RESNET", "MNIST" ,"Linf")
    using_projection( "RESNET", "MNIST" ,"L0")
    using_projection( "RESNET", "MNIST" ,"L1")
    using_projection( "RESNET", "MNIST" ,"L2")

    using_projection( "RESNET", "CIFAR10" ,"L0")
    using_projection( "RESNET", "CIFAR10" ,"L1")
    using_projection( "RESNET", "CIFAR10" ,"L2")
    using_projection( "RESNET", "CIFAR10" ,"Linf")


