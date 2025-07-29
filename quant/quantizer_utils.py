import torch
import os
from typing import Dict, Any, Optional
from .quant_layer import QuantModule, UniformAffineQuantizer
from .quant_block import BaseQuantBlock
from .adaptive_rounding import AdaRoundQuantizer


def save_quantizer_params(model, save_path: str):
    """
    Save delta and alpha parameters from all quantizers in the model.
    
    Args:
        model: The quantized model
        save_path: Path to save the quantizer parameters
    """
    quantizer_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            # Save weight quantizer parameters
            if hasattr(module, 'weight_quantizer'):
                wq = module.weight_quantizer
                if isinstance(wq, AdaRoundQuantizer):
                    quantizer_params[f"{name}.weight_quantizer.alpha"] = wq.alpha.data.clone()
                if hasattr(wq, 'delta') and wq.delta is not None:
                    quantizer_params[f"{name}.weight_quantizer.delta"] = wq.delta.clone()
                if hasattr(wq, 'zero_point') and wq.zero_point is not None:
                    quantizer_params[f"{name}.weight_quantizer.zero_point"] = wq.zero_point.clone()
            
            # Save activation quantizer parameters
            if hasattr(module, 'act_quantizer'):
                aq = module.act_quantizer
                if hasattr(aq, 'delta') and aq.delta is not None:
                    quantizer_params[f"{name}.act_quantizer.delta"] = aq.delta.clone()
                if hasattr(aq, 'zero_point') and aq.zero_point is not None:
                    quantizer_params[f"{name}.act_quantizer.zero_point"] = aq.zero_point.clone()
        
        elif isinstance(module, BaseQuantBlock):
            # Handle block-level quantizers if any
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, QuantModule):
                    full_name = f"{name}.{sub_name}"
                    # Save weight quantizer parameters
                    if hasattr(sub_module, 'weight_quantizer'):
                        wq = sub_module.weight_quantizer
                        if isinstance(wq, AdaRoundQuantizer):
                            quantizer_params[f"{full_name}.weight_quantizer.alpha"] = wq.alpha.data.clone()
                        if hasattr(wq, 'delta') and wq.delta is not None:
                            quantizer_params[f"{full_name}.weight_quantizer.delta"] = wq.delta.clone()
                        if hasattr(wq, 'zero_point') and wq.zero_point is not None:
                            quantizer_params[f"{full_name}.weight_quantizer.zero_point"] = wq.zero_point.clone()
                    
                    # Save activation quantizer parameters
                    if hasattr(sub_module, 'act_quantizer'):
                        aq = sub_module.act_quantizer
                        if hasattr(aq, 'delta') and aq.delta is not None:
                            quantizer_params[f"{full_name}.act_quantizer.delta"] = aq.delta.clone()
                        if hasattr(aq, 'zero_point') and aq.zero_point is not None:
                            quantizer_params[f"{full_name}.act_quantizer.zero_point"] = aq.zero_point.clone()
    
    # Save the quantizer parameters
    torch.save(quantizer_params, save_path)
    print(f"Quantizer parameters saved to: {save_path}")
    print(f"Saved {len(quantizer_params)} quantizer parameters")


def load_quantizer_params(model, load_path: str):
    """
    Load delta and alpha parameters into all quantizers in the model.
    
    Args:
        model: The quantized model
        load_path: Path to load the quantizer parameters from
    
    Returns:
        bool: True if loading was successful, False otherwise
    """
    if not os.path.exists(load_path):
        print(f"Quantizer parameters file not found at: {load_path}")
        return False
    
    try:
        quantizer_params = torch.load(load_path, map_location='cpu')
        loaded_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, QuantModule):
                # Load weight quantizer parameters
                if hasattr(module, 'weight_quantizer'):
                    wq = module.weight_quantizer
                    alpha_key = f"{name}.weight_quantizer.alpha"
                    delta_key = f"{name}.weight_quantizer.delta"
                    zp_key = f"{name}.weight_quantizer.zero_point"
                    
                    if isinstance(wq, AdaRoundQuantizer) and alpha_key in quantizer_params:
                        wq.alpha.data = quantizer_params[alpha_key].to(wq.alpha.device)
                        loaded_count += 1
                    
                    if delta_key in quantizer_params:
                        wq.delta = quantizer_params[delta_key].to(wq.delta.device if hasattr(wq.delta, 'device') else 'cpu')
                        loaded_count += 1
                    
                    if zp_key in quantizer_params:
                        wq.zero_point = quantizer_params[zp_key].to(wq.zero_point.device if hasattr(wq.zero_point, 'device') else 'cpu')
                        loaded_count += 1
                
                # Load activation quantizer parameters
                if hasattr(module, 'act_quantizer'):
                    aq = module.act_quantizer
                    delta_key = f"{name}.act_quantizer.delta"
                    zp_key = f"{name}.act_quantizer.zero_point"
                    
                    if delta_key in quantizer_params:
                        aq.delta = quantizer_params[delta_key].to(aq.delta.device if hasattr(aq.delta, 'device') else 'cpu')
                        loaded_count += 1
                    
                    if zp_key in quantizer_params:
                        aq.zero_point = quantizer_params[zp_key].to(aq.zero_point.device if hasattr(aq.zero_point, 'device') else 'cpu')
                        loaded_count += 1
            
            elif isinstance(module, BaseQuantBlock):
                # Handle block-level quantizers if any
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        full_name = f"{name}.{sub_name}"
                        
                        # Load weight quantizer parameters
                        if hasattr(sub_module, 'weight_quantizer'):
                            wq = sub_module.weight_quantizer
                            alpha_key = f"{full_name}.weight_quantizer.alpha"
                            delta_key = f"{full_name}.weight_quantizer.delta"
                            zp_key = f"{full_name}.weight_quantizer.zero_point"
                            
                            if isinstance(wq, AdaRoundQuantizer) and alpha_key in quantizer_params:
                                wq.alpha.data = quantizer_params[alpha_key].to(wq.alpha.device)
                                loaded_count += 1
                            
                            if delta_key in quantizer_params:
                                wq.delta = quantizer_params[delta_key].to(wq.delta.device if hasattr(wq.delta, 'device') else 'cpu')
                                loaded_count += 1
                            
                            if zp_key in quantizer_params:
                                wq.zero_point = quantizer_params[zp_key].to(wq.zero_point.device if hasattr(wq.zero_point, 'device') else 'cpu')
                                loaded_count += 1
                        
                        # Load activation quantizer parameters
                        if hasattr(sub_module, 'act_quantizer'):
                            aq = sub_module.act_quantizer
                            delta_key = f"{full_name}.act_quantizer.delta"
                            zp_key = f"{full_name}.act_quantizer.zero_point"
                            
                            if delta_key in quantizer_params:
                                aq.delta = quantizer_params[delta_key].to(aq.delta.device if hasattr(aq.delta, 'device') else 'cpu')
                                loaded_count += 1
                            
                            if zp_key in quantizer_params:
                                aq.zero_point = quantizer_params[zp_key].to(aq.zero_point.device if hasattr(aq.zero_point, 'device') else 'cpu')
                                loaded_count += 1
        
        print(f"Successfully loaded {loaded_count} quantizer parameters from: {load_path}")
        return True
        
    except Exception as e:
        print(f"Error loading quantizer parameters: {e}")
        return False


def get_quantizer_params_summary(model):
    """
    Get a summary of all quantizer parameters in the model.
    
    Args:
        model: The quantized model
    
    Returns:
        dict: Summary of quantizer parameters
    """
    summary = {
        'weight_quantizers': {},
        'activation_quantizers': {},
        'total_weight_quantizers': 0,
        'total_activation_quantizers': 0,
        'total_alpha_params': 0,
        'total_delta_params': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            # Weight quantizer summary
            if hasattr(module, 'weight_quantizer'):
                wq = module.weight_quantizer
                wq_info = {
                    'type': type(wq).__name__,
                    'n_bits': wq.n_bits,
                    'has_alpha': isinstance(wq, AdaRoundQuantizer),
                    'has_delta': hasattr(wq, 'delta') and wq.delta is not None,
                    'has_zero_point': hasattr(wq, 'zero_point') and wq.zero_point is not None
                }
                summary['weight_quantizers'][name] = wq_info
                summary['total_weight_quantizers'] += 1
                
                if wq_info['has_alpha']:
                    summary['total_alpha_params'] += 1
                if wq_info['has_delta']:
                    summary['total_delta_params'] += 1
            
            # Activation quantizer summary
            if hasattr(module, 'act_quantizer'):
                aq = module.act_quantizer
                aq_info = {
                    'type': type(aq).__name__,
                    'n_bits': aq.n_bits,
                    'has_delta': hasattr(aq, 'delta') and aq.delta is not None,
                    'has_zero_point': hasattr(aq, 'zero_point') and aq.zero_point is not None
                }
                summary['activation_quantizers'][name] = aq_info
                summary['total_activation_quantizers'] += 1
                
                if aq_info['has_delta']:
                    summary['total_delta_params'] += 1
    
    return summary 