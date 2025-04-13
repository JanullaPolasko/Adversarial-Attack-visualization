import os
import sys
import torch
import torch.optim as optim
import numpy as np
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from datapath import my_path
from adversarial_utils import compute_attack
import subprocess
from adversarial_utils import load_model_eval
from plot_adversarials import plot_adversarials

def run_adversarial_attacks(attack_type='L_inf', dataset='CIFAR10', model_type='CONV', device=None, run= 0 ):
    '''
    This function runs adversarial attacks on a neural network model based on the specified attack type, dataset, and model configuration. it also save the advesarial example to path()

    Parameters:
    - attack_type (str, optional): The type of adversarial attack to perform. Valid values are 'L_inf', 'L2', 'L1', 'L0'. Default is 'L_inf'.
    - dataset (str, optional): The dataset to use for the attack. Valid datasets include 'CIFAR10', 'SVHN', 'FMNIST', 'MNIST'. Default is 'CIFAR10'.
    - model_type (str, optional): The type of model to use. Valid values include 'CONV', 'RESNET'. Default is 'CONV'.
    - device (str, optional): The device on which the model will run. Can be 'cuda' for GPU or 'cpu'. Default is None, which uses the first available GPU or CPU.
    - run (int, optional): The specific run of the experiment. This is used to load a specific checkpoint for the experiment. Default is 0.
    - conf (float, optional): Confidence threshold for filtering adversarial examples based on model confidence. Default is 0.

    Returns:
    - succes_rate (float): The success rate of the attack, calculated based on the number of successful adversarial examples.
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # MODEL INFO
    model_class_name,  model = load_model_eval(dataset, model_type, run, device= device)
    optimizer = optim.Adam(model.parameters())
    pretrained_model =  my_path() + f'/networks/{dataset}_{model_class_name}/{dataset}_run_{run}.pth'
    print('model is saved:', pretrained_model)

    # ATTACK CONFIGURATION
    subprocess.run(['mkdir', '-p', f"{my_path()}/adversarials/{dataset}_{model_class_name}"])

    if attack_type == 'L_inf':
        stp, r_in = (0.06, 10) if dataset == 'CIFAR10' or dataset == 'SVHN' else (0.16, 100)
        eps_range = np.arange(0.01, stp, 0.01)

        succes_rate = []
        for e in eps_range:
            print('attack Linf with', e)
            output_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/Linf_{dataset}_{model_class_name}_{run}_eps_{e:2.2f}.pkl'
            sr = compute_attack(
                        dataset, model, 'Linf', optimizer, output_path, device, eps=e, eps_step=0.001, 
                        max_iter=1000, num_rand_init=r_in,  num_inputs=256, batch=100, high_conf= True)
            succes_rate.append(sr)
    
    elif attack_type == 'L2':
        bin_s, it = (30, 200) if dataset == 'CIFAR10' or dataset == 'SVHN' else (50, 1000)
        output_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/L2_{dataset}_{model_class_name}_{run}.pkl'
        succes_rate =compute_attack(
            dataset, model,  'L2', optimizer, output_path, device, max_iter=it, bin_s_steps=bin_s, 
            batch=128, init_const=0.001, num_inputs=256*8, high_conf= True)
    
    elif attack_type == 'L1':
        bin_s, it = (30, 200) if dataset == 'CIFAR10' or dataset == 'SVHN' else (50, 1000)
        output_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/L1_{dataset}_{model_class_name}_{run}.pkl'
        succes_rate = compute_attack(
            dataset, model,  'L1', optimizer, output_path, device, max_iter=it, bin_s_steps=bin_s, 
            batch=128, init_const=0.001, num_inputs=256*8, high_conf= True )
    
    elif attack_type == 'L0':
        output_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/L0_{dataset}_{model_class_name}_{run}.pkl'
        succes_rate = compute_attack(
            dataset, model, 'L0', optimizer, output_path, device, num_inputs=256*8, 
            max_iter=50, precision=10, high_conf= True )
    
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return succes_rate

if __name__ == "__main__":
    #example of creating AEs with plotting
    run_adversarial_attacks(attack_type='L0', dataset="MNIST", model_type="CONV", run=0)
    plot_adversarials(dataset_name='CIFAR10', network_type='CONV', attack='L0',eps=0.02, n_show=6, run = 0)

