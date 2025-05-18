import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from torch import nn
from adversarial_utils import load_adversarials
from datasets import load_data
from networks import get_dataset_mapping
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import cvxpy as cp

def get_layers(model, pretrained = None):
    """
    Dynamically extracts the leaf modules of the model and constructs corresponding layer indices and names.
    
    Returns:
        layers (list of int): A list of layer indices.
        fl (bool): A flag indicating whether the activations should be flattened. 
        layer_names (list of str): A list of string names for each layer. 
        leaf_modules (list of nn.Module): A list of the collected leaf modules (excluding the skipped types).
    """
    if pretrained == None:
        dataset_mapping = get_dataset_mapping()
        pretrained = False
        for _, _, model_class_mapping, _, pretrained_bool  in dataset_mapping:
            if model == model_class_mapping:
                pretrained = pretrained_bool
                break
    leaf_modules = []

    if pretrained:
        layer_names, layers = [], []
        i = 0
    else:
        layer_names, layers = ['Input'], [0,] 
        i = 1

    for module in model.modules():
        if not list(module.children()):
            if isinstance(module, (nn.Flatten, nn.Dropout,nn.BatchNorm2d , nn.BatchNorm1d, nn.CrossEntropyLoss, nn.Upsample ,nn.ZeroPad2d)):
                leaf_modules.append(module)
                i+=1
                continue  # skip Flatten, Dropout and BatchNorm
            leaf_modules.append(module)
            layers.append(i)
            layer_names.append(module.__class__.__name__)
            i+=1

    has_conv = any(isinstance(m, nn.Conv2d) for m in leaf_modules)
    fl = not has_conv  # Flatten if there are Conv layers

    return layers, fl, layer_names, leaf_modules


def pca_reduction(data_activs, x_activs, sample_size=1000, variance = 0.95):
    """
    Perform PCA dimensionality reduction on the activation data, retaining components that explain the % of the variance.

    Args:
    data_activs (np array) The activation data for the original dataset. It should be a 2D array with shape (n_samples, n_features).     
    x_activs (np.ndarray) The activation data for the adversarial examples. It should also be a 2D array with shape (n_samples, n_features).    
    sample_size (int, default=1000 ): The size of the mini-batch (sample) from `data_activs` to estimate the variance. This sample is used to estimate the number of PCA components to retain, based on the cumulative explained variance.  A larger sample size may result in more accurate estimates but requires more computation.
    variance (int, default = 0.95): How many variance should preserve after the PCA reduction
    Returns:
    data_activs (np.ndarray): The transformed activation data for the original dataset after PCA reduction, with reduced dimensions.    
    x_activs (np.ndarray): The transformed activation data for the adversarial examples after PCA reduction, with reduced dimensions.
    """
        
    # Reshape the data
    n_train = data_activs.shape[0]
    n_adv   = x_activs.shape[0]
    data_activs = data_activs.reshape(n_train, -1)
    x_activs    = x_activs.reshape(n_adv, -1)

    # Step 1: Take a small sample of the data (mini-batch) to estimate variance
    sample_indices = np.random.choice(n_train, sample_size, replace=False)
    data_sample = data_activs[sample_indices]
    
    # Step 2: Fit PCA on the small sample to estimate the number of components
    pca_temp = PCA(svd_solver='randomized')
    pca_temp.fit(data_sample)
    
    # Step 3: Calculate the cumulative variance and find the number of components to retain 95% variance
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_variance >= variance) + 1  # Get the number of components
 
    # Step 4: Fit the final PCA model on the whole dataset with the estimated number of components
    pca = PCA(n_components=n_components_95, svd_solver="randomized")
    data_activs = pca.fit_transform(data_activs)
    x_activs = pca.transform(x_activs)
    
    return data_activs, x_activs


# def activations(model, data, leaf_modules, layers, flat=True, batch_size=64, fp16=False):
#     """
#     Memory-efficient activation extraction that returns the array directly.
#     Processes data in batches and aggressively cleans up intermediate tensors.
#     """
#     device = next(model.parameters()).device
#     dtype = torch.float16 if fp16 else torch.float32
#     activation_list = []

#     # Convert data to tensor if needed
#     if isinstance(data, np.ndarray):
#         data = torch.from_numpy(data).to(device=device, dtype=dtype)
#     else:
#         data = data.to(device=device, dtype=dtype)

#     # Handle input layer special case
#     if layers == 0:
#         if flat:
#             return data.flatten(start_dim=1).cpu().numpy()
#         return data.cpu().numpy()

#     # Create the minimal computation graph needed
#     model_segment = torch.nn.Sequential(*leaf_modules[1:layers+1])
#     model_segment = model_segment.to(device).eval()

#     with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
#         for batch_start in range(0, len(data), batch_size):
#             batch_data = data[batch_start:batch_start + batch_size]
            
#             # Process batch
#             activations = model_segment(batch_data)
            
#             # Convert and store results
#             if flat:
#                 activations = activations.flatten(start_dim=1)
                
#             # Move to CPU immediately and convert to numpy
#             activation_list.append(activations.cpu().numpy())
            
#             # Aggressive cleanup
#             del batch_data, activations
#             torch.cuda.empty_cache()
#             gc.collect()

#     # Combine all batches while keeping memory usage controlled
#     final_array = np.concatenate(activation_list, axis=0)
#     del activation_list
#     gc.collect()
    
#     return final_array



def activations(model, data, leaf_modules, layers =0,  flat=True, Resnet = False, batch=64):
    """
    Extracts activations from the specified layer(s) using the provided modules.
    
    For non-ResNet models (Resnet=False), a sequential forward pass is used. The function
    applies the modules in `leaf_modules` sequentially (where index 0 is the input).
    For ResNet-like models (Resnet=True), a hook-based method is used. 
    the parameter `module_index` specifies which module to hook.
    
    Args:
        model (nn.Module): The neural network model.
        data (numpy.ndarray or torch.Tensor): The input data.
        leaf_modules: the leaf modules (with an Identity at index 0 for input).
        layers (int or list of int): For non-ResNet mode, the layer index (or indices) for which
            activations are to be extracted. Use 0 for the input, 1 for output after the first module, etc.
        module_index (int): For ResNet mode, the index of the desired module in leaf_modules.
        flat (bool): Whether to flatten the activations (except for the batch dimension).
        Resnet (bool): If True, use hook-based extraction (for ResNet-like models); if False, use a sequential forward pass.
        batch (int): Batch size for processing.
    
    Returns:
        numpy.ndarray: The concatenated activations over all batches.
            (For non-ResNet, if only one layer is requested, a single NumPy array is returned.)
    """
   
    torch.cuda.empty_cache()
    device = next(model.parameters()).device
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(device)

    if not Resnet:
        if isinstance(layers, int):
            layers = [layers] 

        len_model_layers = len(leaf_modules)
        act = [None] * len(layers)

        # PROCESS DATA IN BATCHES FOR EFFICIENCY
        for batch_ind in range((data.shape[0] - 1) // batch + 1):
            data_batch = data[batch_ind * batch:(batch_ind + 1) * batch]
            if flat:
                data_batch = torch.flatten(data_batch, 1)

            activ = [data_batch] + [None] * len_model_layers

            # FORWARD PASS THROUGH EACH LAYER
            for lay in range(len_model_layers):
                activ[lay + 1] = leaf_modules[lay](activ[lay].clone())

            if batch_ind > 0:
                for index, layer in enumerate(layers):
                    act[index] = np.concatenate((act[index], np.array(torch.flatten(activ[layer], 1).cpu().detach().numpy())))
            else:
                for index, layer in enumerate(layers):
                    act[index] = np.array(torch.flatten(activ[layer], 1).cpu().detach().numpy())
        return act[0]

    else:
        module_index = layers
        activations_list = []
        # If module_index is 0, simply return the input
        if module_index == 0:
            activations_list = []
            with torch.no_grad():
                for i in range(0, data.size(0), batch):
                    batch_data = data[i : i + batch]
                    if flat:
                        batch_data = torch.flatten(batch_data, 1)
                    activations_list.append(batch_data.detach().cpu().numpy())
            return np.concatenate(activations_list, axis=0)
        
        # Get the hook module directly.
        hook_module = leaf_modules[module_index]

        # Ensure only one activation is recorded per forward pass.
        activation_recorded = False
        # Hook function to capture the output.
        def hook_fn(module, input, output):
            nonlocal activation_recorded
            if activation_recorded:
                return
            activation_recorded = True
            out = output
            if flat:
                out = torch.flatten(out, 1)
            activations_list.append(out.detach().cpu().numpy())

        hook_handle = hook_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            for i in range(0, data.size(0), batch):
                activation_recorded = False
                batch_data = data[i : i + batch]
                if flat:
                    batch_data = torch.flatten(batch_data, 1)
                _ = model(batch_data)  # Trigger forward hook.
                torch.cuda.empty_cache()

        hook_handle.remove()
        return np.concatenate(activations_list, axis=0)

def get_neigh(points, k, x_train):
    """
    Finds the indices of the k-nearest neighbors for a set of points in the training data.

    Args:
    - points (numpy array): The set of points to find neighbors for.
    - k (int): The number of neighbors to find.
    - x_train (numpy array): The training data.

    Returns:
    - nb_ind (numpy array): The indices of the k-nearest neighbors for each point.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree').fit(x_train.reshape(x_train.shape[0], -1))
    points = points.reshape(points.shape[0], -1)
    _, nb_ind = nbrs.kneighbors(points)
    return nb_ind

def convex_projection(X, y, max_slsqp_tries=100):
    """
    Exact convex projection of y onto the convex hull of columns of X
    using constrained QP (SLSQP), with fallback to NNLS if it fails.
    
    Returns:
      - p: projected point (D,)
      - w: weights (k,)
    """
    D, k = X.shape
    # objective: mean-squared error
    fun = lambda w: np.sum((X.dot(w) - y)**2) / D

    # constraints: sum w_i = 1, w_i >= 0
    cons = (
        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq','fun': lambda w: w})
    x0 = np.full(k, 1.0/k)

    for attempt in range(1, max_slsqp_tries+1):
        res = minimize(fun, x0, method='SLSQP', constraints=cons,
                       options={'ftol':1e-9, 'maxiter':100, 'disp':False})
        if res.success:
            w = res.x
            p = X.dot(w)
            return p, w
        else:
            # update x0 
            x0 = res.x if res.x is not None else x0
            #print(f"SLSQP attempt {attempt} failed: {res.message}")

    # Fallback to NNLS-based projection
    print("Falling back to NNLS projection")
    #  min ||X w - y||_2 subject to w >= 0
    w, _ = nnls(X, y)
    # Normalize to make it a convex combination
    total = w.sum()
    if total > 0:
        w = w / total

    p = X.dot(w)
    return p, w

             
def project_points(points, k, x_train, y_train, classes, cls=None, batch=True):
    '''
    points: body ktorych projekcie chceme najst. Ma mat tvar (N, *)
    k: pocet susedov pouzitych na vypocet aproximacie manifoldu
    cls: poradove cislo pozadovanej triedy (musi byt int)
    '''
    #TAKE ONLY CLASS 
    if cls is not None:
        assert cls in range(len(classes))
        mask = np.argmax(y_train, axis=1) == cls
        x_train = x_train[mask]

    # KNN 
    flattened_x_train = x_train.reshape(x_train.shape[0], -1)
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree').fit(flattened_x_train)
    points = points.reshape(points.shape[0], -1)  

    projections, distances = [], []

    if batch:
        #find knn for all points, base  are them in x 
        _, nb_ind = nbrs.kneighbors(points)
        base = flattened_x_train[nb_ind]
        #kazdy bod premietneme na konvexny obal jeho k susedov a ulozi sa vzdialenost toho bodu
        for i in range(base.shape[0]):
            X = base[i].T
            p, _ = convex_projection(X, points[i])
            distances.append(np.linalg.norm(points[i] - p))
            projections.append(p)
    else:
        for i in range(points.shape[0]):
            _, nb_ind = nbrs.kneighbors(points[i:i + 1])
            base = flattened_x_train[nb_ind[0]]
            X = base.T
            p, x = convex_projection(X, points[i])
            distances.append(np.linalg.norm(points[i] - p))
            projections.append(p)

    return projections, distances

def get_coefs(points, k, x_train, y_train, classes, cls=None):
    """
    Computes the convex coefficients for a set of points in relation to their nearest neighbors.

    Args:
    - points (numpy array): The points for which to calculate the convex coefficients.
    - k (int): The number of neighbors used to compute the coefficients.
    - x_train (numpy array): The training data.
    - y_train (numpy array): The training labels.
    - classes (list): List of all possible classes.
    - cls (int, optional): The class to focus on. If None, all classes are considered.

    Returns:
    - nb_ind (numpy array): The indices of the nearest neighbors.
    - coefs (numpy array): The convex coefficients corresponding to each point.
    """

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    points_flat = points.reshape(points.shape[0], -1)

    #TAKE ONLY CLASS 
    if cls is not None:
        assert cls in range(len(classes))
        mask = np.argmax(y_train, axis=1) == cls
        x_train_flat = x_train_flat[mask]

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
    nbrs.fit(x_train_flat)
    _, nb_ind = nbrs.kneighbors(points_flat)

    #EXTRACKT KNN FROM TRAIN DATA TO EVERY POINT
    base = x_train_flat[nb_ind]
    
    n_points = base.shape[0]
    coefs = np.empty((n_points, k))
    # Compute convex coefficients for each point
    for i in range(n_points):
        X = base[i].T  # (features, k)
        _, coefs[i] = convex_projection(X, points_flat[i])

    return nb_ind, coefs

def targets_matrix(attacks, dataset, model_type, how_many = 3, eps = 0.12):
    """
    Creates a confusion matrix based on adversarial examples.

    Parameters:
    - attacks (str): Type of attack ('Linf', 'L2', etc.).
    - dataset (str): The dataset name (e.g., 'MNIST').
    - model_type (str): The model type used (e.g., 'RESNET').

    Returns:
    - matrix (list): Confusion matrix for misclassifications.
    - max_misclassifications (list): List of the most misclassified label pairs (optional).
    """
    maximum_miss = dict()
    original, original_labels, x_test_adv, y_test_adv = load_adversarials(attacks, dataset, model_type, eps=eps)
    _, _, _, classes = load_data(dataset)

    for label in range(len(original_labels)):
        if (original_labels[label], y_test_adv[label]) in maximum_miss.keys():
            maximum_miss[original_labels[label], y_test_adv[label]]+=1
        else:
             maximum_miss[original_labels[label], y_test_adv[label]] = 1
    
    maximum_miss = sorted(maximum_miss.items(), key=lambda item: item[1], reverse=True )
    return [item[0] for item in maximum_miss[:how_many]], maximum_miss

