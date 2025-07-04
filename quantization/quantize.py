def quantize_model(model, quantization_type='dynamic'):
    """
    Quantize the given model to reduce RAM usage and FLOPs.
    
    Parameters:
        model (Union[tf.keras.Model, torch.nn.Module]): The model to quantize.
        quantization_type (str): The type of quantization to apply ('dynamic' or 'static').
        
    Returns:
        Union[tf.keras.Model, torch.nn.Module]: The quantized model.
    """
    if isinstance(model, tf.keras.Model):
        if quantization_type == 'dynamic':
            # Apply dynamic quantization
            quantized_model = tf.quantization.quantize(model, 
                                                         input_quantization=tf.quantization.QuantizationConfig(
                                                             dtype=tf.qint8),
                                                         output_quantization=tf.quantization.QuantizationConfig(
                                                             dtype=tf.qint8))
            return quantized_model
        elif quantization_type == 'static':
            # Apply static quantization
            # This requires a representative dataset for calibration
            raise NotImplementedError("Static quantization is not implemented yet.")
        else:
            raise ValueError("Unsupported quantization type. Use 'dynamic' or 'static'.")

    elif isinstance(model, torch.nn.Module):
        if quantization_type == 'dynamic':
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(model, 
                                                                   {torch.nn.Linear}, 
                                                                   dtype=torch.qint8)
            return quantized_model
        elif quantization_type == 'static':
            # Apply static quantization
            raise NotImplementedError("Static quantization is not implemented yet.")
        else:
            raise ValueError("Unsupported quantization type. Use 'dynamic' or 'static'.")
    
    else:
        raise TypeError("Unsupported model type for quantization (neither TensorFlow/Keras nor PyTorch).")


def analyze_quantization_effects(original_model, quantized_model, dataset):
    """
    Analyze the effects of quantization on model performance and energy efficiency.
    
    Parameters:
        original_model (Union[tf.keras.Model, torch.nn.Module]): The original model.
        quantized_model (Union[tf.keras.Model, torch.nn.Module]): The quantized model.
        dataset (np.ndarray): The dataset used for evaluation.
        
    Returns:
        Dict[str, float]: A dictionary containing performance metrics.
    """
    # Evaluate original model
    original_flops, original_inference_time = estimate_flops(original_model, nb_epochs=1, dataset=dataset)
    
    # Evaluate quantized model
    quantized_flops, quantized_inference_time = estimate_flops(quantized_model, nb_epochs=1, dataset=dataset)
    
    performance_metrics = {
        "original_flops": original_flops,
        "quantized_flops": quantized_flops,
        "original_inference_time": original_inference_time,
        "quantized_inference_time": quantized_inference_time,
        "flops_reduction": (original_flops - quantized_flops) / original_flops * 100,
        "inference_time_reduction": (original_inference_time - quantized_inference_time) / original_inference_time * 100,
    }
    
    return performance_metrics


def print_quantization_report(metrics):
    """
    Print a report on the effects of quantization.
    
    Parameters:
        metrics (Dict[str, float]): The performance metrics from quantization analysis.
    """
    print("\n📉 Quantization Effects Report:")
    print(f"Original FLOPs: {metrics['original_flops']:,}")
    print(f"Quantized FLOPs: {metrics['quantized_flops']:,}")
    print(f"FLOPs Reduction: {metrics['flops_reduction']:.2f}%")
    print(f"Original Inference Time: {metrics['original_inference_time']:.4f} seconds")
    print(f"Quantized Inference Time: {metrics['quantized_inference_time']:.4f} seconds")
    print(f"Inference Time Reduction: {metrics['inference_time_reduction']:.2f}%")