import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import random
import time
import hubconf  # noqa: F401
import copy
import matplotlib.pyplot as plt
from math import sqrt
import collections

import pandas as pd
from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)
from data.imagenet import build_imagenet_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate_model(val_loader, model,fp_model, device=None, print_freq=100,gamma=None,beta=None):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_fp= AverageMeter('Acc@1_fp', ':6.2f')
    top5_fp= AverageMeter('Acc@5_fp', ':6.2f')
    top1_aligned= AverageMeter('Acc@1_aligned', ':6.2f')
    top5_aligned= AverageMeter('Acc@5_aligned', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    fp_model.eval()
    data=[]

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        output_fp=fp_model(images)
        '''for i in range(len(output.mean(dim=0))):
            data.append([output.mean(dim=0)[i].item(),output.std(0)[i].item(),2,2,output_fp.mean(dim=0)[i].item(),output_fp.std(0)[i].item()])'''

        

        mean_x=output.mean()
        #mean_y=output_fp.mean(dim=0)
        std_x=output.std() + 1e-6
        #norm_output = (output - mean_x) / std_x
        #std_y=output_fp.std(0)
        #gamma=std_y/(std_x+1e-6)
        #beta=mean_y-gamma*mean_x
        output_aligned=gamma * output + beta #(output-mean_x+beta)/(gamma*torch.sqrt(std_x))
        #

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure accuracy and record loss
        acc1_fp, acc5_fp = accuracy(output_fp, target, topk=(1, 5))
        top1_fp.update(acc1_fp[0], images.size(0))
        top5_fp.update(acc5_fp[0], images.size(0))

        # measure accuracy and record loss
        acc1_aligned, acc5_aligned = accuracy(output_aligned, target, topk=(1, 5))
        top1_aligned.update(acc1_aligned[0], images.size(0))
        top5_aligned.update(acc5_aligned[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
    '''columns=["q_mean","q_std","a_bit","w_bit","f_mean","f_std"]

    df=pd.DataFrame(data=data,columns=columns)
    df.to_csv("dataset1.csv",index=False)'''

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(' * Acc@1_fp {top1.avg:.3f} Acc@5_fp {top5.avg:.3f}'.format(top1=top1_fp, top5=top5_fp))
    print(' * Acc@1_aligned {top1.avg:.3f} Acc@5_aligned {top5.avg:.3f}'.format(top1=top1_aligned, top5=top5_aligned))
    return top1.avg, top1_fp.avg#, top1_aligned.avg

def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='model name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='/mimer/NOBACKUP/groups/naiss2025-22-91/imagenet', type=str, help='path to ImageNet data')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=2, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=2, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')
    args = parser.parse_args()

    seed_all(args.seed)
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    # load model
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()
    fp_model = copy.deepcopy(cnn)
    fp_model.cuda()
    fp_model.eval()

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    fp_model.set_quant_state(False, False)
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    qnn.disable_network_output_quantization()
    print('the quantized model is below!')
    print(qnn)
    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
                lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu, 
                lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c)


    '''init weight quantizer'''
    set_weight_quantize_params(qnn)

    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)
            # %%
    # Start calibration
    #recon_model(qnn, fp_model)


    
    #torch.save(qnn.state_dict(), "model_full_state1.pth")
    #import sys
    #sys.exit(0)
weights=torch.load("/mimer/NOBACKUP/groups/naiss2025-22-91/ali/change_metrics/model_full_state.pth")

qnn.set_quant_state(weight_quant=True, act_quant=True)

# Load the saved state_dict
checkpoint_state_dict = torch.load("/mimer/NOBACKUP/groups/naiss2025-22-91/ali/change_metrics/model_full_state.pth")

def map_checkpoint_to_qnn(checkpoint_dict, qnn_model):
    """
    Map checkpoint keys to qnn model structure and set delta and alpha values.
    Checkpoint keys: model.layer1.0.conv1.weight_quantizer.alpha
    Qnn structure: qnn.model.layer1[0].conv1.weight_quantizer.delta
    """
    print("Mapping checkpoint keys to qnn model structure...")
    
    # Create a mapping from checkpoint keys to qnn model paths
    key_mapping = {}
    
    # First, let's understand the qnn model structure
    qnn_state_dict = qnn_model.state_dict()
    print(f"Qnn model has {len(qnn_state_dict)} parameters")
    
    # Create a mapping from checkpoint keys to qnn keys
    for checkpoint_key in checkpoint_dict.keys():
        if 'weight_quantizer.alpha' in checkpoint_key or 'weight_quantizer.delta' in checkpoint_key or 'act_quantizer.delta' in checkpoint_key:
            # Convert checkpoint key format to qnn key format
            # checkpoint: model.layer1.0.conv1.weight_quantizer.alpha
            # qnn: model.layer1.0.conv1.weight_quantizer.delta
            
            # Extract the base path (everything before the quantizer part)
            if 'weight_quantizer.alpha' in checkpoint_key:
                base_path = checkpoint_key.replace('.weight_quantizer.alpha', '')
                param_name = 'alpha'
            elif 'weight_quantizer.delta' in checkpoint_key:
                base_path = checkpoint_key.replace('.weight_quantizer.delta', '')
                param_name = 'delta'
            elif 'act_quantizer.delta' in checkpoint_key:
                base_path = checkpoint_key.replace('.act_quantizer.delta', '')
                param_name = 'delta'
            else:
                continue
            
            # Find corresponding qnn key
            qnn_key = None
            for qnn_k in qnn_state_dict.keys():
                if base_path in qnn_k and 'quantizer' in qnn_k:
                    if 'weight_quantizer' in checkpoint_key and 'weight_quantizer' in qnn_k:
                        if param_name == 'alpha':
                            # For alpha, we need to find the AdaRoundQuantizer alpha parameter
                            if 'alpha' in qnn_k:
                                qnn_key = qnn_k
                                break
                        elif param_name == 'delta':
                            if 'delta' in qnn_k:
                                qnn_key = qnn_k
                                break
                    elif 'act_quantizer' in checkpoint_key and 'act_quantizer' in qnn_k:
                        if 'delta' in qnn_k:
                            qnn_key = qnn_k
                            break
            
            if qnn_key:
                key_mapping[checkpoint_key] = qnn_key
                print(f"Mapping: {checkpoint_key} -> {qnn_key}")
            else:
                print(f"No mapping found for: {checkpoint_key}")
    
    # Now set the values in qnn model
    updated_count = 0
    for checkpoint_key, qnn_key in key_mapping.items():
        if checkpoint_key in checkpoint_dict and qnn_key in qnn_state_dict:
            try:
                # Get the value from checkpoint
                checkpoint_value = checkpoint_dict[checkpoint_key]
                
                # Set the value in qnn model
                with torch.no_grad():
                    # Navigate to the specific module and set the parameter
                    module_path = qnn_key.split('.')
                    current_module = qnn_model
                    
                    # Navigate through the module hierarchy
                    for i, path_part in enumerate(module_path[:-1]):
                        if path_part.isdigit():
                            # Handle list indexing like layer1[0]
                            current_module = current_module[int(path_part)]
                        else:
                            current_module = getattr(current_module, path_part)
                    
                    # Set the parameter value
                    param_name = module_path[-1]
                    if hasattr(current_module, param_name):
                        param = getattr(current_module, param_name)
                        if isinstance(param, torch.nn.Parameter):
                            param.data = checkpoint_value.to(param.device)
                        else:
                            setattr(current_module, param_name, checkpoint_value)
                        updated_count += 1
                        print(f"Updated {qnn_key} with value from {checkpoint_key}")
                    else:
                        print(f"Parameter {param_name} not found in module {qnn_key}")
                        
            except Exception as e:
                print(f"Error updating {qnn_key}: {e}")
    
    print(f"Successfully updated {updated_count} parameters")
    return updated_count

# Map and set the checkpoint values
updated_params = map_checkpoint_to_qnn(checkpoint_state_dict, qnn)
print(f"Updated {updated_params} quantizer parameters from checkpoint")

# Alternative direct method for setting quantizer parameters
def set_quantizer_params_directly(checkpoint_dict, qnn_model):
    """
    Directly iterate through qnn model modules and set quantizer parameters.
    This is more reliable than key mapping.
    """
    print("Setting quantizer parameters directly...")
    updated_count = 0
    
    for name, module in qnn_model.named_modules():
        # Handle weight quantizers
        if hasattr(module, 'weight_quantizer'):
            wq = module.weight_quantizer
            checkpoint_key_alpha = name + '.weight_quantizer.alpha'
            checkpoint_key_delta = name + '.weight_quantizer.delta'
            
            # Set alpha if it exists in checkpoint
            if checkpoint_key_alpha in checkpoint_dict:
                if hasattr(wq, 'alpha'):
                    with torch.no_grad():
                        wq.alpha.data = checkpoint_dict[checkpoint_key_alpha].to(wq.alpha.device)
                        updated_count += 1
                        print(f"Updated {checkpoint_key_alpha}")
            
            # Set delta if it exists in checkpoint
            if checkpoint_key_delta in checkpoint_dict:
                if hasattr(wq, 'delta'):
                    with torch.no_grad():
                        wq.delta = checkpoint_dict[checkpoint_key_delta].to(wq.delta.device)
                        updated_count += 1
                        print(f"Updated {checkpoint_key_delta}")
        
        # Handle activation quantizers
        if hasattr(module, 'act_quantizer'):
            aq = module.act_quantizer
            checkpoint_key_delta = name + '.act_quantizer.delta'
            
            # Set delta if it exists in checkpoint
            if checkpoint_key_delta in checkpoint_dict:
                if hasattr(aq, 'delta'):
                    with torch.no_grad():
                        aq.delta = checkpoint_dict[checkpoint_key_delta].to(aq.delta.device)
                        updated_count += 1
                        print(f"Updated {checkpoint_key_delta}")
    
    print(f"Directly updated {updated_count} quantizer parameters")
    return updated_count

# Also try the direct method
direct_updated = set_quantizer_params_directly(checkpoint_state_dict, qnn)
print(f"Direct method updated {direct_updated} parameters")

def verify_quantizer_params(checkpoint_dict, qnn_model):
    """
    Verify that quantizer parameters were set correctly by checking a few samples.
    """
    print("Verifying quantizer parameters...")
    verification_count = 0
    
    for name, module in qnn_model.named_modules():
        if hasattr(module, 'weight_quantizer') or hasattr(module, 'act_quantizer'):
            # Check weight quantizer
            if hasattr(module, 'weight_quantizer'):
                wq = module.weight_quantizer
                checkpoint_key_alpha = name + '.weight_quantizer.alpha'
                checkpoint_key_delta = name + '.weight_quantizer.delta'
                
                if checkpoint_key_alpha in checkpoint_dict and hasattr(wq, 'alpha'):
                    checkpoint_alpha = checkpoint_dict[checkpoint_key_alpha]
                    current_alpha = wq.alpha.data
                    if torch.allclose(checkpoint_alpha.to(current_alpha.device), current_alpha, atol=1e-6):
                        verification_count += 1
                        print(f"✓ {checkpoint_key_alpha} verified")
                    else:
                        print(f"✗ {checkpoint_key_alpha} mismatch")
                
                if checkpoint_key_delta in checkpoint_dict and hasattr(wq, 'delta'):
                    checkpoint_delta = checkpoint_dict[checkpoint_key_delta]
                    current_delta = wq.delta
                    if torch.allclose(checkpoint_delta.to(current_delta.device), current_delta, atol=1e-6):
                        verification_count += 1
                        print(f"✓ {checkpoint_key_delta} verified")
                    else:
                        print(f"✗ {checkpoint_key_delta} mismatch")
            
            # Check activation quantizer
            if hasattr(module, 'act_quantizer'):
                aq = module.act_quantizer
                checkpoint_key_delta = name + '.act_quantizer.delta'
                
                if checkpoint_key_delta in checkpoint_dict and hasattr(aq, 'delta'):
                    checkpoint_delta = checkpoint_dict[checkpoint_key_delta]
                    current_delta = aq.delta
                    if torch.allclose(checkpoint_delta.to(current_delta.device), current_delta, atol=1e-6):
                        verification_count += 1
                        print(f"✓ {checkpoint_key_delta} verified")
                    else:
                        print(f"✗ {checkpoint_key_delta} mismatch")
    
    print(f"Verified {verification_count} quantizer parameters")
    return verification_count

# Verify the parameters
verified_count = verify_quantizer_params(checkpoint_state_dict, qnn)
print(f"Verified {verified_count} quantizer parameters")

def summarize_quantizer_params(checkpoint_dict, qnn_model):
    """
    Summarize what quantizer parameters were found and what was set.
    """
    print("\n=== Quantizer Parameters Summary ===")
    
    # Count checkpoint parameters
    checkpoint_alpha_count = sum(1 for k in checkpoint_dict.keys() if 'weight_quantizer.alpha' in k)
    checkpoint_delta_count = sum(1 for k in checkpoint_dict.keys() if 'weight_quantizer.delta' in k)
    checkpoint_act_delta_count = sum(1 for k in checkpoint_dict.keys() if 'act_quantizer.delta' in k)
    
    print(f"Checkpoint contains:")
    print(f"  - Weight quantizer alpha parameters: {checkpoint_alpha_count}")
    print(f"  - Weight quantizer delta parameters: {checkpoint_delta_count}")
    print(f"  - Activation quantizer delta parameters: {checkpoint_act_delta_count}")
    
    # Count qnn model quantizers
    qnn_weight_quantizers = 0
    qnn_act_quantizers = 0
    qnn_alpha_params = 0
    qnn_delta_params = 0
    
    for name, module in qnn_model.named_modules():
        if hasattr(module, 'weight_quantizer'):
            qnn_weight_quantizers += 1
            wq = module.weight_quantizer
            if hasattr(wq, 'alpha'):
                qnn_alpha_params += 1
            if hasattr(wq, 'delta'):
                qnn_delta_params += 1
        
        if hasattr(module, 'act_quantizer'):
            qnn_act_quantizers += 1
            aq = module.act_quantizer
            if hasattr(aq, 'delta'):
                qnn_delta_params += 1
    
    print(f"Qnn model contains:")
    print(f"  - Weight quantizers: {qnn_weight_quantizers}")
    print(f"  - Activation quantizers: {qnn_act_quantizers}")
    print(f"  - Alpha parameters: {qnn_alpha_params}")
    print(f"  - Delta parameters: {qnn_delta_params}")
    
    # Show sample checkpoint keys
    print(f"\nSample checkpoint keys:")
    alpha_keys = [k for k in checkpoint_dict.keys() if 'weight_quantizer.alpha' in k][:5]
    delta_keys = [k for k in checkpoint_dict.keys() if 'weight_quantizer.delta' in k][:5]
    act_delta_keys = [k for k in checkpoint_dict.keys() if 'act_quantizer.delta' in k][:5]
    
    for key in alpha_keys:
        print(f"  {key}")
    for key in delta_keys:
        print(f"  {key}")
    for key in act_delta_keys:
        print(f"  {key}")

# Generate summary
summarize_quantizer_params(checkpoint_state_dict, qnn)


print("\n=== Fine-tuning final affine layer (gamma, beta) ===")
from torch.optim import Adam

# Forward once to get output shapes
with torch.no_grad():
    sample_output = qnn(cali_data.to(device))
C = sample_output.shape[1]  # num channels (assuming [B, C] or [B, C, H, W])
gamma = nn.Parameter(torch.ones(1, C).to(device))
beta = nn.Parameter(torch.zeros(1, C).to(device))
lr=1e-2
# Optimizer
optimizer = Adam([gamma, beta], lr=lr)
loss_fn = nn.MSELoss()  # Or KLDivLoss with softmax if more suitable
loss_ce=nn.CrossEntropyLoss()
qnn.eval()
loss_list=[]
fp_model.eval()
for epoch in range(400):  # You can tune this
    optimizer.zero_grad()

    with torch.no_grad():
        q_output = qnn(cali_data.to(device))  # quantized output
        #target=cali_target.to(device)
        #mean = q_output.mean(0)
        #std = q_output.std(0) + 1e-6
        #norm_output = (q_output - mean) / std
        fp_output = fp_model(cali_data.to(device))     # full-precision output

# Assume shape [B, C]; reshape gamma and beta if needed
    mean_x=q_output.mean()
    #mean_y=output_fp.mean(dim=0)
    std_x=q_output.std() + 1e-6
    #output_aligned=(q_output-mean_x+beta)/(gamma*torch.sqrt(std_x))

    output_aligned = gamma * q_output + beta
    loss = loss_fn(output_aligned, fp_output)#+loss_ce(output_aligned,target)

    loss.backward()
    loss_list.append(loss.item())
    optimizer.step()

    if epoch % 20 == 0 or epoch == 199:
        print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

print("=> Finished affine fine-tuning")

plt.figure(figsize=(8, 5))
plt.plot(loss_list)
plt.title("Loss Curve (Post-Training Optimization)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()

# Save the plot with a descriptive name
plt.savefig("loss_plateau_gamma_beta_optimization_{lr}_new_method.png".format(lr=lr), dpi=300)
print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
                                                        validate_model(test_loader, qnn,fp_model=fp_model,gamma=gamma,beta=beta)))