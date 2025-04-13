import numpy as np
import torch
from adversarial_utils import load_model_eval, load_data_for_art, load_adversarials
from tqdm import tqdm
from adversarial_utils import load_model_eval, load_data_for_art, load_adversarials
from proximity_utils import activations, get_neigh, get_coefs, project_points, get_layers, activations_orignal
from tqdm import tqdm
import gc
from datapath import my_path
import pickle
import subprocess
from networks import get_dataset_mapping


def compute_method_ratio(dataset, model_type, attacks, orig_class, pred_class, k=100, run = 0, save = True):
    #counting nearest neighbours
    torch.cuda.empty_cache()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # MODEL INFO
    _,  model = load_model_eval(dataset, model_type, run, device= device)

    #ART LOADING FOR
    (x_train, y_train), (_, _), _, _ = load_data_for_art(dataset, batch_size_train=6000, batch_size_test=1000)
    
    eps = 0.12 if dataset in ['MNIST', 'FMNIST'] else 0.04 #decide epsilon in Linf
    original, original_labels, x_test_adv, y_test_adv = load_adversarials(attacks, dataset, model_type, eps= eps)


    #WE NEED TO TREAT RESNET DIFFERENTLY (skip connection and capacity)
    if model_type == 'RESNET' :
        sample_size = 4000
        if len(x_train) > sample_size:
            indices = np.random.choice(len(x_train), sample_size, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]

    #DATA FROM 2 CLASSES
    train_mask = (y_train == orig_class) | (y_train == pred_class) 
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    # those adversarials, which start from orig_class and are predicted as pred_class
    adv_mask = (original_labels == orig_class)
    adv_x = x_test_adv[adv_mask]
    adv_y = y_test_adv[adv_mask]

    # Adversarial examples predicted as pred_class
    valid_adv_mask = (adv_y == pred_class )
    adv_x = adv_x[valid_adv_mask]
    adv_y = adv_y[valid_adv_mask]
    if len(adv_y) == 0:
        raise ValueError('No valid adversarials for the specified classes.')


    #PRETRAINED BOOL
    dataset_mapping = get_dataset_mapping()
    pretrained = False 
    for dataset_name, m_type, model_class, num_classes, pretrained_bool in dataset_mapping:
        if dataset_name == dataset and m_type == model_type:
            pretrained = pretrained_bool
            break

    #GET LAYERS
    layers, fl, _, leaf_modules = get_layers(model, pretrained )


    orig_count = np.zeros((len(layers), adv_x.shape[0]))
    pred_count = np.zeros((len(layers), adv_x.shape[0]))

    for lay in tqdm(range(len(layers))):
        torch.cuda.empty_cache()
        #get activation in training space for each layer
        train_activs = activations(model, x_train,leaf_modules= leaf_modules,layers= layers[lay], pretrained=pretrained, flat= fl)
        adv_activs = activations(model, adv_x, leaf_modules= leaf_modules,layers= layers[lay], pretrained=pretrained, flat =fl)

        #knn for every adv in training space
        nb = get_neigh(adv_activs, k, train_activs)
        found_labels = y_train[nb]

        orig_count[lay, :] = np.sum(found_labels == orig_class, axis=1)
        pred_count[lay, :] = np.sum(found_labels == pred_class, axis=1)
        
        del  train_activs, adv_activs, nb, found_labels
        gc.collect()

    
    #SAVING
    orig_avg = np.average(orig_count, axis=1)
    pred_avg = np.average(pred_count, axis=1)
    orig_std =  np.std(orig_count, axis=1)
    pred_std = np.std(pred_count, axis=1)

    # Save the results into a dictionary (ratio_method)
    ratio_method = {'orig_avg': orig_avg, 'pred_avg': pred_avg, 'orig_std': orig_std, 'pred_std': pred_std}

    if save:
        subprocess.run(['mkdir', '-p', f"{my_path()}/distances/ratio"])
        filename = my_path() + f'/distances/ratio/net_{model_type}_{dataset}_attack_{attacks}_ratio_method.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(ratio_method, f)
        print('file saved to:',filename)

    return ratio_method


def compute_method_projection(dataset, model_type, attacks, orig_class, pred_class, k=50, run =0, save = True):
    # Computing Distances to Class-Specific Manifolds

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # MODEL INFO
    _,  model = load_model_eval(dataset, model_type, run, device= device)
    #ART LOADING FOR
    (x_train, y_train), (x_test, y_test), input_shape, classes = load_data_for_art(dataset, batch_size_train=60000, batch_size_test=10000)
    eps = 0.12 if dataset in ['MNIST', 'FMNIST'] else 0.04 #decide epsilon in Linf
    original, original_labels, x_test_adv, y_test_adv = load_adversarials(attacks, dataset, model_type, eps= eps)


    #WE NEED TO TREAT RESNET DIFFERENTLY (skip connection and capacity)
    if model_type == 'RESNET' :
        sample_size = 1000
        if len(x_train) > sample_size:
            indices = np.random.choice(len(x_train), sample_size, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]

    # Convert data to torch tensors on GPU
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test_adv = torch.from_numpy(x_test_adv).float().to(device)
    y_test_adv = torch.from_numpy(y_test_adv).to(device)
    original_labels = torch.from_numpy(original_labels).to(device)


    #DATA FROM 2 CLASSES
    orig_mask = y_train == orig_class  
    x_orig_cls = x_train[orig_mask]
    pred_mask = y_train == pred_class  
    x_pred_cls = x_train[pred_mask]

    #MASK COMPUTATION
    or_mask = (original_labels == orig_class)  
    x_adv = x_test_adv[or_mask]
    y_adv = y_test_adv[or_mask]

    # are predicted as pred_class
    pr_mask = y_adv == pred_class
    if not pr_mask.any():
        raise ValueError('No valid adversarials for the specified classes.')
    x_adv = x_adv[pr_mask]

    #PRETRAINED BOOL
    dataset_mapping = get_dataset_mapping()
    pretrained = False 
    for dataset_name, m_type, model_class, num_classes, pretrained_bool in dataset_mapping:
        if dataset_name == dataset and m_type == model_type:
            pretrained = pretrained_bool
            break
    #GET LAYERS
    layers, fl, _, leaf_modules = get_layers(model, pretrained)


    #MATRIX FOR DIST
    x_train_flat = x_train.view(x_train.size(0), -1).cpu().numpy()
    n_samples = x_adv.shape[0]
    n_layers = len(layers)
    dist_to_orig = np.zeros((n_layers, n_samples))
    dist_to_pred = np.zeros((n_layers, n_samples))
    counts = np.zeros((n_layers, 3))

    # RESHAPE TO 1D (N, H*W)  (layer, sample, flattened_input_size)
    layer_projections = np.zeros((n_layers, n_samples, x_train_flat.shape[1]))

    for lay in tqdm(range(n_layers)): 
        torch.cuda.empty_cache()

        data_activs = activations(model, x_train,leaf_modules= leaf_modules, layers= layers[lay], Resnet=pretrained, flat= fl)
        x_activs = activations(model, x_adv, leaf_modules= leaf_modules, layers= layers[lay], Resnet=pretrained, flat =fl)
        
        # print('activation1 done')
        # GET COEFICIENT ALPHA AND KNN -optimization problem
        nb_is, ks = get_coefs(x_activs, k, data_activs, y_train, classes)
        base = x_train_flat[nb_is] 
        # projection calculation (i  sample index, j is ngh index, k feature indes)
        projections = np.einsum('ijk,ij->ik', base, ks)
        layer_projections[lay] = projections

        # Reshape projections to input to imsge again (n_samples, flattened) --> (n_samples, C, H, W)
        #ako keby to boli nove obrazky z adv x, sby sa podobali na trenovacie vzorky danej triedy
        #Ak by sa adversariálny príklad opravil tak, aby sa podobal na trénovacie dáta, akú triedu by model priradil?
        projection_x = torch.from_numpy(projections).float().to(device).reshape(-1, *input_shape)

         #CHECK COEF ALPHA SUM
        sum_ks = np.sum(ks, axis=1)
        outlier_mask = (sum_ks > 1.1) | (sum_ks < 0.9)
        if np.any(outlier_mask):
            print(f"Outlier sums found: {sum_ks[outlier_mask]}")

         #GET PREDICTION FOR PROJECTION
        with torch.no_grad():
            otpt_proj = model(projection_x).detach().cpu().numpy()
        preds = np.argmax(otpt_proj, axis=1)

        #RESULT WHERE INPUT GOES (ORIG, PRED, ELSE)
        counts[lay, 0] = np.sum(preds == orig_class)
        counts[lay, 1] = np.sum(preds == pred_class)
        counts[lay, 2] = len(preds) - counts[lay, 0] - counts[lay, 1]

        # COMPUTE DISTANCE TO MANIFOLDS
        _, dist_to_orig[lay] = project_points(projections, k, x_orig_cls.cpu().numpy(), y_train.cpu().numpy(), classes)
        _, dist_to_pred[lay] = project_points(projections, k, x_pred_cls.cpu().numpy(), y_train.cpu().numpy(), classes)

        del data_activs, x_activs, nb_is, ks, base, projections, projection_x, otpt_proj
        torch.cuda.empty_cache()
        gc.collect()
    
    #SAVING
    orig_avg = np.average(dist_to_orig, axis=1)
    pred_avg = np.average(dist_to_pred, axis=1)
    orig_std =  np.std(dist_to_orig, axis=1)
    pred_std = np.std(dist_to_pred, axis=1)

    projection_method = {'orig_avg': orig_avg, 'pred_avg': pred_avg, 'orig_count': orig_std, 'pred_count': pred_std, 'layer_projections': layer_projections}
    if save:
        subprocess.run(['mkdir', '-p', f"{my_path()}/distances/projected"])
        filename = my_path() + f'/distances/projected/net_{model_type}_attack_{attacks}_orig_{orig_class}_pred_{pred_class}_distance.pkl'
        with open(filename, 'wb') as f:
            pickle.dump([orig_avg, pred_avg, layer_projections], f)
        print('file saved to:',filename)
    
    return projection_method
