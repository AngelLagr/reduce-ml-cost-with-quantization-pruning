# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import entropy as scipy_entropy
from ptflops import get_model_complexity_info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def compute_weight_importance_torch(model, dataset, skip_last=True):
    model.eval()
    importance_list = []
    if isinstance(dataset, torch.Tensor):
        current_input = dataset.detach().clone().float()
    else:
        current_input = torch.tensor(dataset, dtype=torch.float32)
    
    linear_layers = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    skip_last = False if not skip_last else len(linear_layers) >= 2
    last_layer_name = linear_layers[-1] if skip_last else None

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            if skip_last and name == last_layer_name:
                continue
            with torch.no_grad():
                z = layer(current_input)
                activation = torch.relu(z) if hasattr(layer, 'activation') else z
                importance = torch.abs(layer.weight) * torch.mean(activation, dim=0).unsqueeze(1)
                importance_list.append((abs(importance.numpy()).T, layer.weight.detach().numpy(), name))
                current_input = activation

    return importance_list

def compute_weight_importance_tf(model, dataset, skip_last=True):
    importance_list = []
    current_input = dataset.copy()

    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    skip_last = False if not skip_last else len(dense_layers) >= 2
    layers_to_analyze = dense_layers[:-1] if skip_last else dense_layers

    for layer in layers_to_analyze:
        if not skip_last and layer == dense_layers[-1]:
            weights, biases = layer.get_weights()
            importance = np.abs(weights)       
            importance_list.append((abs(importance), weights.T, layer.name))
        else:
            weights, biases = layer.get_weights()
            z = np.dot(current_input, weights) + biases
            activations = layer.activation(z)
            importance = np.abs(weights) * np.mean(activations, axis=0)        
            importance_list.append((abs(importance), weights.T, layer.name))
            current_input = activations.numpy() if isinstance(activations, tf.Tensor) else activations

    return importance_list

def compute_weight_importance(model, dataset, skip_last=True):
    try:
        if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
            return compute_weight_importance_tf(model, dataset, skip_last=skip_last)
    except ImportError:
        pass

    try:
        if isinstance(model, torch.nn.Module):
            return compute_weight_importance_torch(model, dataset, skip_last=skip_last)
    except ImportError:
        pass

    raise TypeError("Unrecognized model type: must be a TensorFlow/Keras or PyTorch model.")

def compute_entropy(normalized_importance):
    return scipy_entropy(normalized_importance + 1e-8)

def compute_topk_coverage(importance_flat, k=0.9):
    sorted_importance = np.sort(importance_flat)[::-1]
    cumsum = np.cumsum(sorted_importance)
    threshold = k * np.sum(sorted_importance)
    topk_count = np.searchsorted(cumsum, threshold) + 1
    return topk_count / len(importance_flat)

def plot_importance_histogram(normalized_importance, entropy_val=None):
    plt.figure(figsize=(10, 4))
    plt.hist(normalized_importance, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Weight Contribution Distribution\nEntropy = {entropy_val:.4f}" if entropy_val is not None else "Weight Contribution Distribution")
    plt.xlabel("Normalized Importance")
    plt.ylabel("Number of Weights")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_report(importance, weights):
    importance_flat = importance.flatten()
    normalized_importance = importance_flat / np.sum(importance_flat)
    ent = compute_entropy(normalized_importance)
    topk_90 = compute_topk_coverage(importance_flat, k=0.9)
    low_contrib = np.sum(importance_flat < 1e-2) / len(importance_flat)

    report = {
        "total_weights": np.prod(weights.shape),
        "nodes": weights.shape[0],
        "entropy": float(ent),
        "effective_weights": float(np.exp(ent)),
        "top_90_percent_contrib": float(topk_90),
        "low_contrib_percentage": float(low_contrib * 100),
    }
    return report, normalized_importance

def print_report(report):
    print("\n📊 Weight Usage Report:")
    print(f"Total number of weights: {report['total_weights']}")
    print(f"Number of nodes (neurons): {report['nodes']}")
    print(f"Entropy (measure of uncertainty): {report['entropy']:.4f}")
    print(f"Effective weights (active weights count): {report['effective_weights']:.4f}")
    print(f"Contribution of top 90% of weights: {report['top_90_percent_contrib'] * 100:.2f}%")
    print(f"Percentage of low weights (<1e-2): {report['low_contrib_percentage']:.2f}%")

def estimate_flops(model, nb_epochs, dataset):
    def get_flops_tf_keras(model: tf.keras.Model) -> int:
        from tensorflow.python.framework import convert_to_constants
        concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype))
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'none'

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops

    nb_samples = len(dataset)

    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        flops_per_sample_forward = get_flops_tf_keras(model)
        flops_training = 2 * flops_per_sample_forward * nb_epochs * nb_samples
        flops_inference = flops_per_sample_forward * nb_samples
        return flops_training, flops_inference

    elif isinstance(model, torch.nn.Module):
        input_shape = (dataset.shape[1],)
        macs, _ = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False, verbose=False)
        flops_per_sample_forward = 2 * macs
        flops_training = 2 * flops_per_sample_forward * nb_epochs * nb_samples
        flops_inference = flops_per_sample_forward * nb_samples
        return flops_training, flops_inference

    else:
        raise TypeError("Unsupported model type for FLOPs estimation (neither TensorFlow/Keras nor PyTorch).")

def print_flops_report(model, nb_epochs, dataset):
    train_flops, infer_flops = estimate_flops(model, nb_epochs, dataset)
    nb_samples = len(dataset)
    print(f"\n🧮 FLOPs Estimation:")
    print(f" - Training ({nb_epochs} epochs, {nb_samples} samples): {train_flops:,} operations")
    print(f" - Inference ({nb_samples} samples): {infer_flops:,} operations")

def show(model, dataset):
    importance_list = compute_weight_importance(model, dataset, skip_last=False)    

    layer_positions = []
    layer_names = []

    input_dim = dataset.shape[1]
    input_layer = [f"Input {i}" for i in range(input_dim)]
    layer_positions.append(input_layer)
    layer_names.append("Input")

    for (importance, weights, name) in importance_list:
        importance = np.mean(importance, axis=0)
        layer = [f"{name}\nN{i}" for i in range(len(importance))]
        layer_positions.append(layer)
        layer_names.append(name)

    _, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    spacing_x = 3
    spacing_y = 1.5

    positions = {}
    for i, layer in enumerate(layer_positions):
        x = i * spacing_x
        total = len(layer)
        for j, label in enumerate(layer):
            y = -j * spacing_y + (total - 1) * spacing_y / 2
            ax.add_patch(patches.Circle((x, y), 0.2, color="lightgrey", zorder=2))
            ax.text(x, y, label.split('\n')[0], ha='center', va='center', fontsize=8)
            positions[(i, j)] = (x, y)

    for i in range(len(layer_positions) - 1):
        src_size = len(layer_positions[i])
        dst_size = len(layer_positions[i + 1])
        weights = importance_list[i][1].T 
        for src in range(min(src_size, weights.shape[0])):
            for dst in range(min(dst_size, weights.shape[1])):
                weight = weights[src][dst]
                color = 'red' if abs(weight) > 1e-1 else 'lightgray'
                linewidth = min(max(abs(weight)*3, 0.2), 3)
                x1, y1 = positions[(i, src)]
                x2, y2 = positions[(i + 1, dst)]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=1)

    plt.title("Visualization of your Neural Network with Weight Importance")
    plt.show()