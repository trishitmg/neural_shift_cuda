import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
class config:
    def __init__(self):
        pass



class NECTR_denoiser(nn.Module):
    def __init__(self, args, device):
        super(NECTR_denoiser, self).__init__()
        self.device = device
        self.model_type = args.model_type  
        self.residual_depth = args.residual_depth 
        self.proj_depth = args.proj_depth 
        self.latent_dim = args.latent_dim 
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

        latent_dim = self.latent_dim

        self.in_layer = nn.Conv2d(in_dim, latent_dim, kernel_size=self.patch_size, padding=self.patch_size//2)

        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(),
            )
            for _ in range(self.residual_depth-1)
        ])
        self.in_proj.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim*4, kernel_size=1, padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(latent_dim*4, latent_dim, kernel_size=1, padding=0),
            )
            for _ in range(self.residual_depth)
        ])

        self.in_norm = nn.ModuleList([
            nn.LayerNorm(self.latent_dim)
            for _ in range(self.residual_depth)
        ])

        if self.model_type == 'rbf':
            return

        self.proj = nn.ModuleList([])
        in_dim = 2*latent_dim if self.model_type=='concat' else latent_dim
        if args.proj_depth>0:
            for i in range(args.proj_depth):
                self.proj.append(
                    nn.ModuleList([
                        nn.Conv2d(in_dim//(2**i), in_dim//(2**(i+1)), kernel_size=1),
                        nn.LayerNorm(in_dim//(2**(i+1))),
                        nn.LeakyReLU(),
                    ])
                )
        print(f"self.proj: {self.proj}")

        in_dim = in_dim//(2**(args.proj_depth))
        if self.output_activation=='control_sigmoid':
            out_dim = args.in_channel*3
        else:
            out_dim = args.in_channel

        self.out = nn.Conv2d(in_dim, out_dim, bias=False, kernel_size=1)

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

        # input layer
        x = self.in_layer(x)

        # residual layers
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        return x
    
    def N_theta(self, x, x_s):
        
        if self.model_type == 'concat':
            # concatinating x and (x_s) shifted x.
            x = torch.cat((x, x_s), dim=1)
       
        for proj in self.proj:
            for i, sub_layer in enumerate(proj):
                if i == 1: ## LayerNorm layer
                    x = x.permute(0, 2, 3, 1)
                    x = sub_layer(x)
                    x = x.permute(0, 3, 1, 2)

                else:
                    x = sub_layer(x)
        
        x = self.out(x)

        # positive output activation
        if self.output_activation == 'exp':
            x = torch.exp(x/self.sigma**2)

        elif self.output_activation=='control_sigmoid':
            k, a, b = torch.chunk(x, dim=1, chunks=3)
            x = self.controlled_sigmoid(k, a, b) + 1e-8

        elif self.output_activation=='sig':
            normalization_factor = -(self.sigma * self.sigma)
            return torch.nn.functional.sigmoid(k/normalization_factor)

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
 
