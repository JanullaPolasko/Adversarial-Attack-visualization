import matplotlib.pyplot as plt
import numpy as np
from datapath import my_path
from adversarial_utils import load_adversarials , load_model_eval
import subprocess
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_data, calculate_mean_std, unnormalize_image
from networks import get_dataset_mapping


def plot_adversarials(dataset_name, network_type, attack='L0', run=0, loss = nn.CrossEntropyLoss(),  eps=0.15, n_show=6, fsizex=23):
    """
    This function visualizes adversarial examples by displaying the original and adversarial images for a specified attack type. The images are displayed in a grid format with the original image in the first row and the adversarial image in the second row.

    Parameters:
    - dataset_name (str): The name of the dataset (e.g., 'CIFAR10', 'MNIST', 'FMNIST', 'SVHN').
    - network_type (str): The type of network used (e.g., 'CONV', 'FC').
    - attack (str, optional): The type of attack used (e.g., 'L0', 'L1', 'L2', 'Linf'). Default is 'L0'.
    - run (int, optional): The run index (default is 0).
    - loss (torch.nn.Module, optional): The loss function used in the model. Default is nn.CrossEntropyLoss().
    - eps (float, optional): The epsilon parameter for the attack. Default is 0.15.
    - n_show (int, optional): The number of examples to display. Default is 6.
    - fsizex (int, optional): Font size for titles and axis labels in the plot. Default is 23.

    Returns:
    - None (the function saves the plot to a file).
    """
    np.set_printoptions(precision=3, suppress=True)
    subprocess.run(['mkdir', '-p', f"{my_path()}/images"])
    attacks = ['L0', 'L1', 'L2', 'Linf']
    attack_names = [r'L$_0$', r'L$_1$', r'L$_2$', r'L$_{\infty}$']
    
    #LOAD MODEL
    _, model = load_model_eval(dataset_name, network_type, run)
    optimizer = optim.Adam(model.parameters())
    _, _, input_shape, classes = load_data(dataset_name)
    
    #SET MIN AND MAX BASED ON THE NORMALIZATION
    mean, std = calculate_mean_std(dataset_name)
    min_pix = (0 - mean) / std
    max_pix = (1 - mean) / std

    if min_pix.ndim == 1:
        min_pix = min_pix.view(1, -1, 1, 1)
        max_pix = max_pix.view(1, -1, 1, 1)   
    min_pix = min_pix.cpu().numpy()
    max_pix = max_pix.cpu().numpy()
    
    classifier = PyTorchClassifier(model=model, clip_values=(min_pix, max_pix), loss=loss, optimizer=optimizer, input_shape=input_shape, nb_classes=len(classes))
    
    #LOAD ADVERSARIAL 
    original,_, x_test_adv, _ = load_adversarials(attack, dataset_name, network_type,  eps=eps, run=0)
    if original is None or x_test_adv is None:
        print("Error: Adversarial examples not loaded correctly!")
        return

    #PREDICTION PROB
    logits_orig = classifier.predict(original)  
    probs_orig = F.softmax(torch.tensor(logits_orig), dim=1).numpy() 
    y_pred_orig = probs_orig.argmax(axis=1)
    confidences_orig = probs_orig.max(axis=1) * 100  

    logits_adv = classifier.predict(x_test_adv)
    probs_adv = F.softmax(torch.tensor(logits_adv), dim=1).numpy()
    y_pred_adv = probs_adv.argmax(axis=1)
    confidences_adv = probs_adv.max(axis=1) * 100  
    
    # PLOTTING 
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(f"{attack_names[attacks.index(attack)]} on dataset {dataset_name} with model {network_type}", fontsize=fsizex)
    for j in range(1, n_show + 1):
        idx = n_show + j
            
        # --- Row 1: Original image ---
        ax_orig = fig.add_subplot(2, n_show, j)
        # UNNORMALIZE
        orig_img_tensor = torch.tensor(original[idx], dtype=torch.float)
        orig_img_tensor = unnormalize_image(orig_img_tensor, dataset_name, split='train', batch_size=64)
        orig_img_np = orig_img_tensor.cpu().numpy()
        if dataset_name in ['CIFAR10', 'SVHN']:
            # For multi-channel images, adjust the axes.
            orig_img_np = np.moveaxis(orig_img_np, 0, -1)
            ax_orig.imshow(orig_img_np, vmin=0, vmax=1)
        else:
            # For grayscale images, use the first channel.
            ax_orig.imshow(orig_img_np[0], cmap='gray', vmin=0, vmax=1)
        ax_orig.axis('off')
        if classes is not None:
            ax_orig.set_title(f"Orig: {classes[y_pred_orig[idx]]} ({confidences_orig[idx]:.1f}%)", fontsize=fsizex - 14)
        else:
            ax_orig.set_title(f"Orig: {y_pred_orig[idx]} ({confidences_orig[idx]:.1f}%)", fontsize=fsizex - 14)
            
        # --- Row 2: Adversarial image ---
        ax_pred = fig.add_subplot(2, n_show, j + n_show)
        # UNNORMALIZE
        adv_img_tensor = torch.tensor(x_test_adv[idx], dtype=torch.float)
        adv_img_tensor = unnormalize_image(adv_img_tensor, dataset_name, split='train', batch_size=64)
        adv_img_np = adv_img_tensor.cpu().numpy()
        if dataset_name in ['CIFAR10', 'SVHN']:
            adv_img_np = np.moveaxis(adv_img_np, 0, -1)
            ax_pred.imshow(adv_img_np, vmin=0, vmax=1)
        else:
            ax_pred.imshow(adv_img_np[0], cmap='gray', vmin=0, vmax=1)
        ax_pred.axis('off')
        if classes is not None:
            ax_pred.set_title(f"Pred: {classes[y_pred_adv[idx]]} ({confidences_adv[idx]:.1f}%)", fontsize=fsizex - 14)
        else:
            ax_pred.set_title(f"Pred: {y_pred_adv[idx]} ({confidences_adv[idx]:.1f}%)", fontsize=fsizex - 14)
        
    #SAVE
    save_path = my_path() + f'/images/{network_type}_{dataset_name}_{attack}_adversarials.png'
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")

def plot_attack_with_diff_everydatasets(attack, sample_idx=0, run=0, eps=0.15, loss = nn.CrossEntropyLoss()):
    """
    This function visualizes adversarial examples for every dataset (MNIST, CIFAR10, etc.) and attack type (L0, L1, L2, Linf). It shows the original, difference, and adversarial images for each dataset in a grid.

    Parameters:
    - attack (str): The attack type to visualize (e.g., 'L0', 'L1', 'L2', 'Linf').
    - sample_idx (int, optional): The index of the sample to visualize. Default is 0.
    - run (int, optional): The run index (default is 0).
    - eps (float, optional): The epsilon parameter for the attack. Default is 0.15.
    - loss (torch.nn.Module, optional): The loss function used in the model. Default is nn.CrossEntropyLoss().

    Returns:
    - None (the function saves the plot to a file).
    """
    dataset_mapping = get_dataset_mapping()
    fig, axs = plt.subplots(2, 6, figsize=(21, 7))

    dataset_index = 0
    for entry in dataset_mapping:
        dataset_name, network_type, model_class, num_classes, pretrained = entry
        if network_type == 'FC':
            continue

        # LOAD MODEL and DATA (your existing code)
        model_class_name, model = load_model_eval(dataset_name, network_type, run)
        optimizer = optim.Adam(model.parameters())
        _, _, input_shape, classes = load_data(dataset_name)
        
        mean, std = calculate_mean_std(dataset_name)
        min_pix = (0 - mean) / std
        max_pix = (1 - mean) / std
        if min_pix.ndim == 1:
            min_pix = min_pix.view(1, -1, 1, 1)
            max_pix = max_pix.view(1, -1, 1, 1)   
        min_pix = min_pix.cpu().numpy()
        max_pix = max_pix.cpu().numpy()
        
        classifier = PyTorchClassifier(model=model, clip_values=(min_pix, max_pix), 
                                    loss=loss, optimizer=optimizer, 
                                    input_shape=input_shape, nb_classes=len(classes))
        
        original,_, x_test_adv, _ = load_adversarials(attack, dataset_name, network_type, eps=eps, run=0)
        
        if sample_idx >= len(original):
            print(f"Sample index {sample_idx} out of range (max {len(original)-1}).")
            return

        # SAMPLE AND PREDICT
        orig_img = original[sample_idx]
        orig_img_batch = np.expand_dims(orig_img, axis=0)
        orig_img_tensor = torch.tensor(orig_img, dtype=torch.float)

        adv_img  = x_test_adv[sample_idx]
        adv_img_batch = np.expand_dims(adv_img, axis=0)
        adv_img_tensor = torch.tensor(adv_img, dtype=torch.float)

        logits_orig = classifier.predict(orig_img_batch)  
        probs_orig = F.softmax(torch.tensor(logits_orig), dim=1).numpy() 
        y_pred_orig = probs_orig.argmax(axis=1)
        confidences_orig = probs_orig.max(axis=1) * 100  

        logits_adv = classifier.predict(adv_img_batch)
        probs_adv = F.softmax(torch.tensor(logits_adv), dim=1).numpy()
        y_pred_adv = probs_adv.argmax(axis=1)
        confidences_adv = probs_adv.max(axis=1) * 100  

        # UNNORMALIZE
        orig_img_tensor = unnormalize_image(orig_img_tensor, dataset_name, split='train', batch_size=64)
        orig_img = orig_img_tensor.cpu().numpy()
        adv_img_tensor = unnormalize_image(adv_img_tensor, dataset_name, split='train', batch_size=64)
        adv_img = adv_img_tensor.cpu().numpy()

        # COMPUTE DIFFERENCE
        diff = np.abs(adv_img - orig_img)
        if dataset_name in ['CIFAR10', 'SVHN']:
            # Color images: adjust channel position and compute 2D diff mask.
            orig_disp = np.moveaxis(orig_img, 0, -1)
            adv_disp  = np.moveaxis(adv_img, 0, -1)
            diff_disp = np.max(diff, axis=0)
            cmap = 'hot_r'
        else:
            # Grayscale images.
            orig_disp = orig_img[0, :, :]
            adv_disp  = adv_img[0, :, :]
            diff_disp = diff[0, :, :]
            cmap = 'hot_r'
        
        # Calculate position in the 2x6 grid.
        row = dataset_index // 2               # two datasets per row
        col_offset = (dataset_index % 2) * 3     # each dataset uses 3 columns

        # Plot Original Image
        axs[row, col_offset].imshow(orig_disp, cmap=('gray' if dataset_name not in ['CIFAR10', 'SVHN'] else None))
        axs[row, col_offset].set_title(f"Original\n{classes[y_pred_orig[0]]} ({confidences_orig[0]:.1f}%)", fontsize = 14)
        axs[row, col_offset].axis('off')

        # Plot Difference Image
        axs[row, col_offset+1].set_title("Difference (Changed Pixels)",  fontsize = 14)
        im = axs[row, col_offset+1].imshow(diff_disp, cmap=cmap)
        axs[row, col_offset+1].text(0.5, 1.2, f"{dataset_name}", transform=axs[row, col_offset+1].transAxes, ha="center", fontsize=19)
        axs[row, col_offset+1].axis('off')
        fig.colorbar(im, ax=axs[row, col_offset+1], fraction=0.046, pad=0.04)

        # Plot Adversarial Image
        axs[row, col_offset+2].imshow(adv_disp, cmap=('gray' if dataset_name not in ['CIFAR10', 'SVHN'] else None))
        axs[row, col_offset+2].set_title(f"Adversarial\n{classes[y_pred_adv[0]]} ({confidences_adv[0]:.1f}%)",  fontsize = 14)
        axs[row, col_offset+2].axis('off')

        dataset_index += 1
    
    #SAVE
    save_path = my_path() + f'/images/Example_Everything_{attack}_adversarials.png'
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")
    plt.show()

def plot_single_attack_Linf_all_eps(dataset_name, network_type, sample_idx=0, run=0, loss=nn.CrossEntropyLoss()):
    """
    This function visualizes adversarial examples generated using the 'Linf' attack for multiple epsilon values. It shows the original image, the difference image (between original and adversarial), and the adversarial image for each epsilon value.

    Parameters:
    - dataset_name (str): The name of the dataset (e.g., 'CIFAR10', 'MNIST').
    - network_type (str): The type of network used (e.g., 'CONV', 'FC').
    - sample_idx (int, optional): The index of the sample to visualize. Default is 0.
    - run (int, optional): The run index (default is 0).
    - loss (torch.nn.Module, optional): The loss function used in the model. Default is nn.CrossEntropyLoss().

    Returns:
    - None (the function saves the plot to a file).
    """
    # Define epsilon range based on dataset parameters.
    if dataset_name in ['CIFAR10', 'SVHN']:
        eps_range= [0.01, 0.03, 0.05]
    else:
        eps_range= [0.01,0.04, 0.08, 0.12, 0.15]


    # LOAD MODEL and DATA
    model_class_name, model = load_model_eval(dataset_name, network_type, run)
    optimizer = optim.Adam(model.parameters())
    _, _, input_shape, classes = load_data(dataset_name)
    
    # SET MIN and MAX pixel values based on normalization.
    mean, std = calculate_mean_std(dataset_name)
    min_pix = (0 - mean) / std
    max_pix = (1 - mean) / std
    if min_pix.ndim == 1:
        min_pix = min_pix.view(1, -1, 1, 1)
        max_pix = max_pix.view(1, -1, 1, 1)
    min_pix = min_pix.cpu().numpy()
    max_pix = max_pix.cpu().numpy()
    
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pix, max_pix),
        loss=loss,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=len(classes)
    )
    
    # Prepare a figure with one row per epsilon (3 columns: original, difference, adversarial).
    n_rows = len(eps_range)
    fig, axs = plt.subplots(n_rows, 3, figsize=(7.5, 2.5* n_rows))
    fig.suptitle(f"{dataset_name}" , fontsize=20)
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    
    # Loop over each epsilon value.
    for i, eps in enumerate(eps_range):
        # Load the adversarial data (this call returns the original as well) for the current epsilon.
        original,_, x_test_adv, y_test_adv = load_adversarials("Linf", dataset_name, network_type,  eps=eps, run=0)
        
        if sample_idx >= len(original):
            print(f"Sample index {sample_idx} out of range for epsilon {eps:.2f}. Skipping this row.")
            continue
        
        # Extract sample images for this epsilon.
        orig_img = original[sample_idx]
        adv_img  = x_test_adv[sample_idx]
        true_label = y_test_adv[sample_idx]
        
        orig_img_batch = np.expand_dims(orig_img, axis=0)
        adv_img_batch  = np.expand_dims(adv_img, axis=0)
        orig_img_tensor = torch.tensor(orig_img, dtype=torch.float)
        adv_img_tensor = torch.tensor(adv_img, dtype=torch.float)
        
        # Compute predictions for the original image.
        logits_orig = classifier.predict(orig_img_batch)
        probs_orig = F.softmax(torch.tensor(logits_orig), dim=1).numpy()
        y_pred_orig = probs_orig.argmax(axis=1)
        confidences_orig = probs_orig.max(axis=1) * 100
        
        # Compute predictions for the adversarial image.
        logits_adv = classifier.predict(adv_img_batch)
        probs_adv = F.softmax(torch.tensor(logits_adv), dim=1).numpy()
        y_pred_adv = probs_adv.argmax(axis=1)
        confidences_adv = probs_adv.max(axis=1) * 100
        
        # Unnormalize images.
        orig_img_tensor = unnormalize_image(orig_img_tensor, dataset_name, split='train', batch_size=64)
        adv_img_tensor  = unnormalize_image(adv_img_tensor, dataset_name, split='train', batch_size=64)
        orig_img = orig_img_tensor.cpu().numpy()
        adv_img = adv_img_tensor.cpu().numpy()
        
        # Compute difference.
        diff = np.abs(orig_img - adv_img)
        
        if dataset_name in ['CIFAR10', 'SVHN']:
            orig_disp = np.moveaxis(orig_img, 0, -1)
            adv_disp  = np.moveaxis(adv_img, 0, -1)
            diff_disp = np.max(diff, axis=0)
        else:
            orig_disp = orig_img[0, :, :]
            adv_disp  = adv_img[0, :, :]
            diff_disp = diff[0, :, :]
        
        # Plot the original image.
        axs[i, 0].imshow(orig_disp, cmap=('gray' if dataset_name not in ['CIFAR10', 'SVHN'] else None))
        axs[i, 0].set_title(f"Original\n{classes[y_pred_orig[0]]} ({confidences_orig[0]:.1f}%)", fontsize=14)
        axs[i, 0].axis('off')
        
        # Plot the difference image.
        axs[i, 1].set_title(f"Difference\n Max Eps: {eps:.2f}", fontsize=14)
        axs[i, 1].axis('off')
        im = axs[i, 1].imshow(diff_disp, cmap='hot_r', vmin=0, vmax=eps_range[-1])
        fig.colorbar(im, ax=axs[i, 1], fraction=0.046, pad=0.04)
        
        # Plot the adversarial image.
        axs[i, 2].imshow(adv_disp, cmap=('gray' if dataset_name not in ['CIFAR10', 'SVHN'] else None))
        axs[i, 2].set_title(f"Adversarial\n{classes[y_pred_adv[0]]} ({confidences_adv[0]:.1f}%)", fontsize=14)
        axs[i, 2].axis('off')
    

    plt.subplots_adjust(wspace=0.44)
    save_path = my_path() + f'/images/Example_{network_type}_{dataset_name}_Linf_all_eps_adversarials.png'
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")
    plt.show()



