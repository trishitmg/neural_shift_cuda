import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import deepinv as dinv
class config:
    def __init__(self):
        pass



class NECTR_denoiser(nn.Module):
    def __init__(self, args, device):
        super(NECTR_denoiser, self).__init__()
        self.device = device
        self.model_type = args.model_type 

        self.scales = getattr(args, 'scales', 4)

        self.sigma = 1
        self.blind = args.blind
        self.patch_size = 2*args.patch_rad + 1
        self.output_activation = args.output_activation

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel
        

        if args.blind:
            in_dim=self.in_channel
        else:
            in_dim=(self.in_channel+1)


        if self.output_activation=='control_sigmoid':
            out_dim = self.in_channel*3
        else:
            out_dim = self.in_channel

        self.unet = dinv.models.UNet(
            in_channels=2*in_dim,
            out_channels=self.in_channel,
            scales=self.scales,  # number of downsampling/upsampling scales
            residual=False,
            cat=True,
            batch_norm=False,
        ).to(device)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.out = nn.Conv2d(self.in_channel, out_dim, bias=False, kernel_size=1)






    def pre_activation(self, x, sigma=None):
        if not self.blind:
            if sigma is not None:
                if isinstance(sigma, torch.Tensor):
                    sigma = sigma
                elif isinstance(sigma, float):
                    sigma = torch.tensor([sigma], dtype=x.dtype, device=x.device)
                
                # Reshape sigma to [batch, 1, 1, 1] then expand to match x's spatial dimensions
                if sigma.dim() == 2 and sigma.shape[1] == 1:
                    # sigma is [batch, 1], reshape to [batch, 1, 1, 1]
                    sigma = sigma.view(sigma.shape[0], 1, 1, 1)
                elif sigma.dim() == 1:
                    # sigma is [batch], reshape to [batch, 1, 1, 1]
                    sigma = sigma.view(sigma.shape[0], 1, 1, 1)
                elif sigma.dim() == 0:
                    # sigma is scalar, reshape to [1, 1, 1, 1]
                    sigma = sigma.view(1, 1, 1, 1)

                sigma_map = sigma.expand(x.shape[0], 1, x.shape[2], x.shape[3])
                x = torch.concat((x, sigma_map), dim=1)
            else:
                raise ValueError('sigma is None, but blind is False')

        
        return x
    
   

    
    def N_theta(self, x, x_s):
        
        if self.model_type == 'concat':
            # concatinating x and (x_s) shifted x.
            x = torch.cat((x, x_s), dim=1)
        x = self.unet(x)
        x = x.permute(0, 2, 3, 1)  # rearranging to match LayerNorm input shape
        x = self.layer_norm(x)  # applying LayerNorm
        x = x.permute(0, 3, 1, 2)  # rearranging back to original shape
        x = self.out(x)

       

        if self.output_activation=='control_sigmoid':
            k, a, b = torch.chunk(x, dim=1, chunks=3)
            x = self.controlled_sigmoid(k, a, b) + 1e-8
            


        return x

    def controlled_sigmoid(self, x, a, b):
        # a(x) = relu/sq(alpha) / ( 1 + exp ( - sigmoid (beta) x)
        return F.softplus(a)/(1+torch.exp(-F.sigmoid(b)*x))

    def forward(self, x, reference=None, sig=None, cache = None, store_cache=False): #cache holds the NN output for each permutation
        U = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        Z = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        if reference is None:
            reference = self.pre_activation(x, sigma=sig)
        else:
            reference = self.pre_activation(reference, sigma=sig)

        padded_img = F.pad(x, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
        padded_reference = F.pad(reference, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
        box = F.pad(torch.ones_like(x, device=x.device, dtype=x.dtype), (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='constant', value=0)

        hold_cache = {}
        count_real = 0
        count_total = 0
        for dx in range(-self.window_rad, self.window_rad + 1):
            for dy in range(1, self.window_rad + 1):
        
                reference_shifted = padded_reference[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]

                if cache is None:
                    weight = self.N_theta(reference, reference_shifted)
                    comp_box = box[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = weight * comp_box
                    if store_cache:
                        key = (dx, dy)
                        hold_cache[key] = weight
                if cache is not None:
                    key = (dx, dy)
                    weight = cache[key]

                v = padded_img[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                U = U + weight * v
                Z = Z + weight
                
                dx_inv = -dx
                dy_inv = -dy
                weight_padded = F.pad(weight, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
                weight = weight_padded[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                comp_box = box[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]

                weight = weight * comp_box
                if torch.isnan(weight).any():
                    print(f"NaN detected in weight for dx {dx} and dy {dy}")
                v = padded_img[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                U = U + weight * v
                Z = Z + weight

        for dx in range(0, self.window_rad + 1):
            for dy in [0]:
        
                reference_shifted = padded_reference[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                
                if cache is None:
                    weight = self.N_theta(reference, reference_shifted)
                    comp_box = box[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                    weight = weight * comp_box
                    if store_cache:
                        key = (dx, dy)
                        hold_cache[key] = weight
                if cache is not None:
                    key = (dx, dy)
                    weight = cache[key]


                v = padded_img[:, :, self.window_rad + dx:self.window_rad + dx + x.shape[2], self.window_rad + dy:self.window_rad + dy + x.shape[3]]
                U = U + weight * v
                Z = Z + weight
                

                if dx == 0 and dy ==0:
                    continue

                dx_inv = -dx
                dy_inv = -dy
                
                weight_padded = F.pad(weight, (self.window_rad, self.window_rad, self.window_rad, self.window_rad), mode='circular')
                weight = weight_padded[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                comp_box = box[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]

                weight = weight * comp_box
                
                v = padded_img[:, :, self.window_rad + dx_inv:self.window_rad + dx_inv + x.shape[2], self.window_rad + dy_inv:self.window_rad + dy_inv + x.shape[3]]
                U = U + weight * v
                Z = Z + weight

        U = U / Z
        if store_cache:
            return U, Z, hold_cache
        else:
            return U,Z


    def SYMM_FORWARD(self, x, reference, sig=None):
        # Find D
        one = torch.ones_like(x, device=x.device, dtype=x.dtype)
        _, D, cache = self.forward(one, reference, sig=sig, store_cache=True)

        #Find one_hat
        one_hat = 1.0/torch.sqrt(D)
        one_hat, _ = self.forward(one_hat, reference, sig=sig, cache=cache)
        one_hat = torch.sqrt(D) * one_hat

        #find one_hat supremum
        one_hat_sup = torch.amax(one_hat, dim=[2,3], keepdim=True)

        normalized_one_hat = one_hat / one_hat_sup

        #find second component
        x_tilde = (one - normalized_one_hat)*x

        #find first component
        x1 = (1/torch.sqrt(D))*x
        x1, _ = self.forward(x1, reference, sig=sig, cache=cache)
        x1 = x1 * torch.sqrt(D)
        x1 = x1/one_hat_sup

        U = x1 + x_tilde
        return U, D
    

    def SYMM_FORWARD_NAIVE(self, x, reference, sig=None):
        # Find D
        one = torch.ones_like(x, device=x.device, dtype=x.dtype)
        _, D = self.forward(one, reference, sig=sig)

        #Find one_hat
        one_hat = 1.0/torch.sqrt(D)
        one_hat, _ = self.forward(one_hat, reference, sig=sig)
        one_hat = torch.sqrt(D) * one_hat

        #find one_hat supremum
        one_hat_sup = torch.amax(one_hat, dim=[2,3], keepdim=True)

        normalized_one_hat = one_hat / one_hat_sup

        #find second component
        x_tilde = (one - normalized_one_hat)*x

        #find first component
        x1 = (1/torch.sqrt(D))*x
        x1, _ = self.forward(x1, reference, sig=sig)
        x1 = x1 * torch.sqrt(D)
        x1 = x1/one_hat_sup

        U = x1 + x_tilde
        return U, D

