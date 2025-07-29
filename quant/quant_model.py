import torch.nn as nn
import torch
from .quant_block import specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn


class IRMLayer(nn.Module):
    """
    Individual Response Modulation (IRM) layer
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Learnable parameters for transformation
        self.alpha = nn.Parameter(torch.ones(1, input_dim))
        self.beta = nn.Parameter(torch.zeros(1, input_dim))
        
    def forward(self, x):
        # Apply individual response modulation
        # x shape: (batch_size, input_dim)
        transformed = self.alpha * x + self.beta
        return transformed


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()

        #self.gamma=torch.nn.Parameter(torch.ones(1,dim))
        #self.beta=torch.nn.Parameter(torch.zeros(1,dim))
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)
<<<<<<< HEAD
        
        # Add IRM layer - will be initialized after first forward pass
        self.irm_layer = None
=======
        # Add IRM (Information Rectification Module)
        self.gamma = nn.Parameter(torch.ones(1000))
        self.beta = nn.Parameter(torch.zeros(1000))
        #self.gamma1 = nn.Parameter(torch.ones(1000))
        #self.beta1 = nn.Parameter(torch.zeros(1000))
>>>>>>> b6fee08384f2622b12d311adf7d5746550a5c4fe

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, nn.BatchNorm2d):
                if prev_quantmodule is not None:
                    prev_quantmodule.norm_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
            
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
<<<<<<< HEAD
        output = self.model(input)
        
        # Initialize IRM layer if not already done
        if self.irm_layer is None:
            # Get the output dimension from the first forward pass
            if output.dim() > 2:
                # For convolutional outputs, flatten to (batch_size, features)
                batch_size = output.size(0)
                output_flat = output.view(batch_size, -1)
                input_dim = output_flat.size(1)
            else:
                input_dim = output.size(1)
            
            self.irm_layer = IRMLayer(input_dim).to(output.device)
        
        # Apply IRM transformation
        if output.dim() > 2:
            # For convolutional outputs, flatten, apply IRM, then reshape back
            batch_size = output.size(0)
            original_shape = output.shape
            output_flat = output.view(batch_size, -1)
            transformed_flat = self.irm_layer(output_flat)
            output = transformed_flat.view(original_shape)
        else:
            output = self.irm_layer(output)
        
=======
        output=self.model(input)
        output = self.gamma * output + self.beta
        #output = self.gamma1 * output + self.beta1
>>>>>>> b6fee08384f2622b12d311adf7d5746550a5c4fe
        return output

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)
        # a_list[0].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
