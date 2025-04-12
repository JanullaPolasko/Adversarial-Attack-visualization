#functions which help to run attack based on parameters
import torch.nn.functional as F
from datapath import my_path
import torch
from torch import nn
import pickle
import numpy as np
from tqdm import tqdm
from networks import get_dataset_mapping
from datasets import calculate_mean_std
from datasets import load_data

from art.attacks.evasion import ProjectedGradientDescent, ElasticNet, CarliniL2Method
from art.estimators.classification import PyTorchClassifier

def compute_attack(dataset, model,  attck, optimizer, filename, device, num_inputs=10000, loss=nn.CrossEntropyLoss(), min_pix=0.0, max_pix=1.0,
                   conf=0.9, bin_s_steps=50, max_iter=1000, precision=10, init_const=0.001, batch=512, max_h=90,
                   max_d=90, beta=0.001, eps=0.1, eps_step=0.01, num_rand_init=200, high_conf=False):
    '''
    Parameters:
    - dataset: string like when loading a dataset from a file, e.g., 'MNIST'
    - model: model class, i.e., architecture
    - pretrained_model: path to the saved dictionary
    - attack: type of attack, valid values are 'L0', 'L1', 'L2', 'Linf'
    - optimizer: Optimizer (e.g., torch.optim.Optimizer), needed when using attack methods
    - filename (str): Path to the file where attack results will be saved (using pickle).
    - device: (e.g., 'cpu' or 'cuda')
    - num_inputs (default=10000): Number of examples used for the attack
    - loss (default = nn.CrossEntropyLoss()): Loss function used in the attack.
    - exclude_incorrect (default=True): Excludes those examples that the model already misclassifies before performing the attack.
    - transform (default = transforms.ToTensor()): Transformation applied to the data.
    - min_pix (default=0.0): Minimum pixel value used for image clipping.
    - max_pix (default=1.0): Maximum pixel value used for image clipping.
    - adv_conf (default=0.95): Confidence threshold used if high_conf=True. Used to filter attack results based on how confident the model is in its prediction after the attack.
    - high_conf (default=False): If True, performs additional filtering of attack results and keeps only those where the model’s confidence is above adv_conf.
    - max_iter (default=1000): Maximum number of iterations (steps) for attack optimization.

    L1 Attack:
    - conf (default=0.0): By adjusting the confidence parameter, you can control the desired confidence level of the adversarial examples.
    - bin_s_steps (default=50): Number of binary search steps used in attacks where this technique is applied.
    - init_const (default=0.001): Initial constant used in optimization methods.
    - batch (default=512): Batch size used for generating adversarial examples.
    - beta (default=0.01): Weight parameter used in the ElasticNet attack, controlling the trade-off between different parts of the loss.

    L2 Attack:
    - max_h (default=90): Maximum number of "halving" steps in the Carlini-L2 attack (decreasing the constant). - Makes the attack more meticulous in finding a minimal perturbation that fools the neural network.
    - max_d (default=90): Maximum number of "doubling" steps in the Carlini-L2 attack (increasing the constant). - Makes the attack faster but potentially results in larger perturbations.

    L0 Attack:
    - precision (default=10): Number of test values per channel used in the iterative pixel attack.

    L_inf Attack:
    - eps (default=0.1): Maximum perturbation (epsilon) for attacks based on norms.
    - eps_step (default=0.01): Step size for iterative attacks, such as Projected Gradient Descent (PGD).
    - num_rand_init (default=200): Number of random initializations for attacks that use this parameter (e.g., PGD with multiple random starting points).

    '''
    
    assert attck in ['L0', 'L1', 'L2', 'Linf']
    torch.cuda.empty_cache()

    #ART LOADING FOR
    _, (x_test, y_test), input_shape, classes = load_data_for_art(dataset, batch_size_train=60000, batch_size_test=10000)
    x_test = x_test[:num_inputs]
    y_test = y_test[:num_inputs] 

    #MIXING
    indices = np.random.permutation(len(x_test))
    x_test = x_test[indices][:num_inputs]
    y_test = y_test[indices][:num_inputs]

    #SET MIN AND MAX BASED ON THE NORMALIZATION
    mean, std = calculate_mean_std(dataset)

    min_pix = (0 - mean) / std
    max_pix = (1 - mean) / std
    min_pix = min_pix.cpu().numpy()
    max_pix = max_pix.cpu().numpy()


    #for 3 dim i need to recalculate it but for 1 dim i can past it like tuple
    if len(min_pix) == 3: 
        min_pix = np.reshape(min_pix, (1, 3, 1, 1))  
        max_pix = np.reshape(max_pix, (1, 3, 1, 1))  

    # SCALE EPS
    if std.numel() > 1:
        scalar_std = std.max().item()
    else:
        scalar_std = std.item()

    eps = eps / scalar_std
    eps_step = eps_step / scalar_std


    classifier = PyTorchClassifier(model=model, clip_values=(min_pix, max_pix), loss=loss, optimizer=optimizer, input_shape=input_shape, nb_classes=len(classes))

    # EXCLUDE INCORRECT PREDICTIONS
    prediction = classifier.predict(x_test)
    init_mask = np.argmax(prediction, axis=1) == y_test
    print(f"Initial correct predictions: {np.sum(init_mask)} / {len(init_mask)} = {(np.sum(init_mask) / len(init_mask)) * 100}")
    x_test = x_test[init_mask]
    y_test = y_test[init_mask]

    # ATTACK SETUP
    print(f"Setting up attack for {attck}...")
    if attck == 'L1':
        attack = ElasticNet(classifier, confidence=conf, binary_search_steps=bin_s_steps, max_iter=max_iter, beta=beta, initial_const=init_const, batch_size=batch, decision_rule='L1')
    elif attck == 'L2':
        attack = CarliniL2Method(classifier, confidence=conf, binary_search_steps=bin_s_steps, max_iter=max_iter, initial_const=init_const, max_halving=max_h, max_doubling=max_d, batch_size=batch)
    elif attck == 'Linf':
        attack = ProjectedGradientDescent(classifier, norm='inf', eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=num_rand_init, batch_size=batch)
    print(f"Attack setup complete: {attck} attack with parameters: conf={conf}, max_iter={max_iter}, batch_size={batch}")
    
    #PERFORM ATTACK
    if attck != 'L0':
        x_test_adv = attack.generate(x=x_test)  # ART attacks work with numpy arrays
        new_mask = np.argmax(classifier.predict(x_test_adv), axis=1) != y_test
    else:
        x_test_adv = iterative_pixel_attack(x_test, y_test, model, input_shape, device, max_iter=max_iter, precision=precision, min_pix=min_pix, max_pix=max_pix, conf=conf).astype(np.float32)
        new_mask = np.argmax(classifier.predict(x_test_adv), axis=1) != y_test

    
    # SUCCESS OF ATTACK
    success_rate = np.sum(new_mask) / len(new_mask)
    x_test_adv = x_test_adv[new_mask]
    y_test_adv = y_test[new_mask]
    original = x_test[new_mask]


    #GETO MODEL PRED FOR ALL CONF IN LIST - IF HIGH CONF TRUE
    conf_result = []
    if high_conf:
        conf_list = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.0]
        logits = classifier.predict(x_test_adv)
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        max_conf = np.max(probabilities, axis=1)
        
        original_sample_count = len(x_test) 
        for conf_value in conf_list:
            conf_mask = max_conf >= conf_value
            success_rate = np.sum(conf_mask) / original_sample_count
            conf_result.append(success_rate)
            print(f"Confidence ≥{conf_value*100:.0f}%: {success_rate*100:.2f}%")

        #SAVE FOR THE LAST CONF = 0
        x_test_adv = x_test_adv[conf_mask]
        original = original[conf_mask]
        y_test_adv = y_test_adv[conf_mask]
        if len(x_test) > 0:
            success_rate = np.sum(conf_mask) / len(x_test)  # Original valid samples count

    print(f"Adversarial success rate: {success_rate * 100:.2f}%")
    # SAVING RESULTS
    print(f"Saving results to: {filename}")
    with open(filename, 'wb') as f:
        pickle.dump([original, x_test_adv, y_test_adv], f)

    return success_rate

def iterative_pixel_attack(x, y, model, input_shape, device, min_pix, max_pix, max_iter=50, precision=10, conf=0):
    """
    Generate adversarial examples by iteratively modifying pixels with maximum gradient.
    
    Args:
        x: Input data (NCHW format)
        y: Target labels (1D array of class indices)
        model: Target model
        input_shape: Shape of input (C, H, W)
        device: Computation device
        min_pix: Minimum pixel value (per channel or global)
        max_pix: Maximum pixel value (per channel or global)
        max_iter: Maximum iterations per sample
        precision: Number of test values per channel
        conf: Minimum confidence required for a successful attack
    
    Returns:
        adversarials: Generated adversarial examples
    """
    adversarials = np.zeros_like(x)
    succ = 0
    n_channels = input_shape[0]
    
    # Generate test values for each channel
    if min_pix.ndim == 1:
        test_values_channels = [
            np.linspace(float(min_pix[c]), float(max_pix[c]), num=precision) 
            for c in range(n_channels)
        ]
        values = np.stack(np.meshgrid(*test_values_channels, indexing='ij'), axis=-1).reshape(-1, n_channels)
    else:
        test_values_channels = [
            np.linspace(float(min_pix[0, c, 0, 0]), float(max_pix[0, c, 0, 0]), num=precision) 
            for c in range(n_channels)
        ]
        channels = np.meshgrid(*test_values_channels, indexing='ij')
        values = np.stack(channels, axis=-1).reshape(-1, n_channels)

    for i in tqdm(range(x.shape[0]), desc="Processing samples"):
        current_adv = x[i].copy()
        changed = np.zeros((x.shape[2], x.shape[3]), dtype=bool)
        target_class = y[i]  # Target class index

        for _ in range(max_iter):
            # Convert to tensor and compute gradients
            inp = torch.tensor(current_adv[np.newaxis], device=device, requires_grad=True, dtype=torch.float32)
            output = model(inp)
            loss = model.loss(output, torch.tensor([target_class], device=device))
            loss.backward()
            
            # Process gradients to find the most impactful pixel
            grad = inp.grad.data.abs().cpu().numpy()
            channel_sum = grad.sum(1).squeeze()
            channel_sum[changed] = -np.inf  # Ignore already changed pixels
            py, px = np.unravel_index(np.argmax(channel_sum), channel_sum.shape)
            changed[py, px] = True

            # Generate candidate perturbations for the selected pixel
            candidates = np.tile(current_adv, (len(values), 1, 1, 1))
            candidates[:, :, py, px] = values  # Set all channels for the pixel

            # Evaluate candidates
            candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(candidate_tensor).cpu().numpy()
            
            # Compute probabilities and determine successful candidates
            probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
            max_probs = np.max(probs, axis=1)
            pred_classes = np.argmax(outputs, axis=1)
            success_mask = (pred_classes != target_class) & (max_probs >= conf)
            success_indices = np.where(success_mask)[0]

            if len(success_indices) > 0:
                # Select best successful candidate (minimizes target class logit)
                best_success_idx = success_indices[np.argmin(outputs[success_indices, target_class])]
                current_adv = candidates[best_success_idx]
                succ += 1
                break
            else:
                # Select candidate that minimizes target class logit
                best_idx = np.argmin(outputs[:, target_class])
                current_adv_candidate = candidates[best_idx]
                # Check if this candidate meets success conditions
                current_pred = pred_classes[best_idx]
                current_conf = max_probs[best_idx]
                if current_pred != target_class and current_conf >= conf:
                    current_adv = current_adv_candidate
                    succ += 1
                    break
                else:
                    current_adv = current_adv_candidate

        adversarials[i] = current_adv

    success_rate = succ / x.shape[0]
    print(f'Attack success rate: {success_rate * 100:.2f}%')
    return adversarials

def load_model_eval(dataset_name, network_type, run , device= None):
    '''
    Loads a pre-trained model for evaluation purposes. The model is selected based on the dataset name and network type.

    Parameters:
    - dataset_name (str): The name of the dataset (e.g., 'MNIST').
    - network_type (str): The type of the model architecture (e.g., 'RESNET').
    - run (int): A specific run identifier (used to load different model checkpoints for the same dataset and network type).
    - device (str, optional): The device on which the model should run (e.g., 'cpu' or 'cuda').

    Returns:
    - model_class_name (str): The name of the model class.
    - model (torch.nn.Module): The loaded PyTorch model, ready for evaluation.
    '''

    
    #MODEL INFORMATION
    dataset_mapping = get_dataset_mapping()
    model_class, pretrained = None, False
    for entry in dataset_mapping:
        if entry[0] == dataset_name and entry[1] == network_type:
            _, _, model_class, _, pretrained = entry
            break

    # MODEL CHECK 
    if model_class is None:
        raise ValueError(f"Invalid dataset '{dataset_name}' or network type '{network_type}'!")

    if pretrained:
        model = model_class()
    else:
        model = model_class(F.cross_entropy)
    model_class_name = model_class.__name__

    pretrained_model = my_path() + f'/networks/{dataset_name}_{model_class_name}/{dataset_name}_run_{run}.pth'
    try:
        #MODEL LOADING
        checkpoint = torch.load(pretrained_model, weights_only=True)
        if "model." in list(checkpoint.keys())[0]:  
            checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint )
        if device != None:
            model.to(device)
        model.eval()

    except FileNotFoundError:
        raise FileNotFoundError(f"Pretrained model not found at: {pretrained_model}")
    return model_class_name,  model


def load_adversarials(attack, dataset, type, eps=0.05, shuffle=True, run=0):
    '''
    Loads adversarial examples from disk based on the specified attack type.

    Parameters:
    - attack (str): The attack type used for generating adversarial examples (e.g., 'Linf', 'L2', 'L0', 'L1').
    - dataset (str): The dataset name (e.g., 'MNIST').
    - type (str): The model type used (e.g., 'RESNET').
    - return_all (bool, optional): Whether to return all adversarial examples (True) or only successful ones (False). Default is True.
    - eps (float, optional): Perturbation magnitude used in the attack (default is 0.07).
    - shuffle (bool, optional): Whether to shuffle the examples before returning them. Default is True.
    - must_fool (bool, optional): If True, only successful adversarial examples (that mislead the model) will be returned.
    - run (int, optional): The specific run of the experiment (used for identifying checkpoint files).

    Returns:
    - original (numpy.ndarray): The original images before the attack.
    - original_labels (numpy.ndarray): The predicted labels for the original images based on the model.
    - x_test_adv (numpy.ndarray): The adversarially modified images.
    - y_test_adv (numpy.ndarray): The labels of the adversarial images.
    '''
    original, x_test_adv, original_labels, y_test_adv = [], [], [], []

    # MODEL INFO
    model_class_name, model = load_model_eval(dataset, type, run)

    # ATTACK TYPE
    if attack == 'Linf':
        file_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/Linf_{dataset}_{model_class_name}_{run}_eps_{eps:2.2f}.pkl'
        with open(file_path, 'rb') as f:
            original, x_test_adv, original_labels = pickle.load(f)

    elif attack in ['L2', 'L0', 'L1']:
        file_path = f'{my_path()}/adversarials/{dataset}_{model_class_name}/{attack}_{dataset}_{model_class_name}_{run}.pkl'
        with open(file_path, 'rb') as f:
            original, x_test_adv, original_labels = pickle.load(f)
    
    print(f"Loading file: {file_path}")
    print(f"Loaded: {len(original)} original, {len(x_test_adv)} adversarial, {len(original_labels)} labels")

    # PREDICT LABELS FOR ADVERSARIAL EXAMPLES
    with torch.no_grad():
        model_output = model(torch.tensor(x_test_adv).to(next(model.parameters()).device))
        y_test_adv = torch.argmax(model_output, dim=1).cpu().numpy()

    # Verifying successful adversarial examples (must fool the model)
    corr_msk = y_test_adv != original_labels

    # Filter only successful adversarial examples (where the model's prediction is incorrect)
    original = original[corr_msk]
    x_test_adv = x_test_adv[corr_msk]
    y_test_adv = y_test_adv[corr_msk]
    original_labels = original_labels[corr_msk]

    if shuffle:
        perm = np.random.permutation(len(original))
        original = original[perm]
        x_test_adv = x_test_adv[perm]
        y_test_adv = y_test_adv[perm]
        original_labels = original_labels[perm]

    return original, original_labels, x_test_adv, y_test_adv



def load_data_for_art(dataset, batch_size_train=64, batch_size_test=64):
    '''
    Loads the dataset for use with the Adversarial Robustness Toolbox (ART). It loads the dataset and returns the training and test data in numpy array format.

    Parameters:
    - dataset (str): The dataset name (e.g., 'MNIST').
    - batch_size_train (int, optional): The batch size for the training dataset (default is 64).
    - batch_size_test (int, optional): The batch size for the test dataset (default is 64).

    Returns:
    - (x_train, y_train) (tuple): The training images and labels as numpy arrays.
    - (x_test, y_test) (tuple): The test images and labels as numpy arrays.
    - input_shape (tuple): The shape of the input data.
    - classes (list): The class labels.

    '''
        
    train_loader, test_loader, input_shape, classes = load_data( dataset, batch_size_train=batch_size_train,batch_size_test=batch_size_test)
    
    # Process training data
    x_train, y_train = [], []
    for data, labels in train_loader:
        x_train.append(data)
        y_train.append(labels)
    x_train = torch.cat(x_train).cpu().numpy()
    y_train = torch.cat(y_train).cpu().numpy()

    # Process test data
    x_test, y_test = [], []
    for data, labels in test_loader:
        x_test.append(data)
        y_test.append(labels)
    x_test = torch.cat(x_test).cpu().numpy()
    y_test = torch.cat(y_test).cpu().numpy()

    return (x_train, y_train), (x_test, y_test), input_shape, classes

