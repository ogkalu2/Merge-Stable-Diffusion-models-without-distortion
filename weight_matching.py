from collections import defaultdict
from re import L
from typing import NamedTuple
import torch
from scipy.optimize import linear_sum_assignment

import jax.numpy as jnp
from jax import random

class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
  """We assume that one permutation cannot appear in two axes of the same weight array."""
  assert num_hidden_layers >= 1
  return permutation_spec_from_axes_to_perm({
      "layer0.weight": ("P_0", None),
      **{f"layer{i}.weight": ( f"P_{i}", f"P_{i-1}")
         for i in range(1, num_hidden_layers)},
      **{f"layer{i}.bias": (f"P_{i}", )
         for i in range(num_hidden_layers)},
      f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
      f"layer{num_hidden_layers}.bias": (None, ),
  })

def sdunet_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in,), f"{name}.bias": (p_out,) }
  norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}
  skip = lambda name, p_in, p_out: {f"{name}": (p_out, p_in, None, None, )}
  
  # Unet Res blocks
  easyblock = lambda name, p_in, p_out: {
  **norm(f"{name}.in_layers.0", p_in),
  **conv(f"{name}.in_layers.2", p_in, f"P_{name}_inner"),
  **dense(f"{name}.emb_layers.1", f"P_{name}_inner2", f"P_{name}_inner3", bias=True),
  **norm(f"{name}.out_layers.0", f"P_{name}_inner4"),
  **conv(f"{name}.out_layers.3", f"P_{name}_inner4", p_out),
  }

  # Text Encoder blocks
  easyblock2 = lambda name, p: {
  **norm(f"{name}.norm1", p),
  **conv(f"{name}.conv1", p, f"P_{name}_inner"),
  **norm(f"{name}.norm2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p),
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
  **norm(f"{name}.norm1", p_in),
  **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
  **norm(f"{name}.norm2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
  **conv(f"{name}.nin_shortcut", p_in, p_out),
  **norm(f"{name}.nin_shortcut", p_out),
  }


  return permutation_spec_from_axes_to_perm({
     #Skipped Layers
     **skip("betas", None, None),
     **skip("alphas_cumprod", None, None),
     **skip("alphas_cumprod_prev", None, None),
     **skip("sqrt_alphas_cumprod", None, None),
     **skip("sqrt_one_minus_alphas_cumprod", None, None),
     **skip("log_one_minus_alphas_cumprods", None, None),
     **skip("sqrt_recip_alphas_cumprod", None, None),
     **skip("sqrt_recipm1_alphas_cumprod", None, None),
     **skip("posterior_variance", None, None),
     **skip("posterior_log_variance_clipped", None, None),
     **skip("posterior_mean_coef1", None, None),
     **skip("posterior_mean_coef2", None, None),
     
     #initial 
     **dense("model.diffusion_model.time_embed.0", None, "P_bg0", bias=True),
     **dense("model.diffusion_model.time_embed.2","P_bg0", "P_bg1", bias=True),
     **conv("model.diffusion_model.input_blocks.0.0", "P_bg2", "P_bg3"),
     
     #input blocks    
     **easyblock("model.diffusion_model.input_blocks.1.0","P_bg4", "P_bg5"),     
     **norm("model.diffusion_model.input_blocks.1.1.norm", "P_bg6"),
     **conv("model.diffusion_model.input_blocks.1.1.proj_in", "P_bg6", "P_bg7"),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q", "P_bg8", "P_bg9", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k", "P_bg8", "P_bg9", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v", "P_bg8", "P_bg9", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0", "P_bg8", "P_bg9", bias=True),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj", "P_bg10", "P_bg11", bias=True),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2", "P_bg12", "P_bg13", bias=True),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q", "P_bg14", "P_bg15", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k", "P_bg16", "P_bg17", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v", "P_bg16", "P_bg17", bias=False),
     **dense("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0", "P_bg18", "P_bg19", bias=True),
     **norm("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1", "P_bg19" ),
     **norm("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2", "P_bg19"),
     **norm("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3", "P_bg19"),
     **conv("model.diffusion_model.input_blocks.1.1.proj_out", "P_bg19", "P_bg20"),

     **easyblock("model.diffusion_model.input_blocks.2.0", "P_bg21","P_bg22"),
     **norm("model.diffusion_model.input_blocks.2.1.norm", "P_bg23"),  
     **conv("model.diffusion_model.input_blocks.2.1.proj_in", "P_bg23", "P_bg24"),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q", "P_bg25", "P_bg26", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k", "P_bg25", "P_bg26", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v", "P_bg25", "P_bg26", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0", "P_bg25","P_bg26", bias=True),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj", "P_bg27","P_bg28", bias=True),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2", "P_bg29","P_bg30", bias=True),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q", "P_bg31", "P_bg32", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k", "P_bg33", "P_bg34", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v", "P_bg33", "P_bg34", bias=False),
     **dense("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0", "P_bg35","P_bg36", bias=True),
     **norm("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1", "P_bg36"),
     **norm("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2", "P_bg36"),
     **norm("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3", "P_bg36"),
     **conv("model.diffusion_model.input_blocks.2.1.proj_out", "P_bg36", "P_bg37"),   

     **conv("model.diffusion_model.input_blocks.3.0.op", "P_bg38", "P_bg39"),
     **easyblock("model.diffusion_model.input_blocks.4.0", "P_bg40","P_bg41"),
     **conv("model.diffusion_model.input_blocks.4.0.skip_connection", "P_bg42","P_bg43"),
     **norm("model.diffusion_model.input_blocks.4.1.norm", "P_bg44"),
     **conv("model.diffusion_model.input_blocks.4.1.proj_in", "P_bg44", "P_bg45"),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q", "P_bg46", "P_bg47", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k", "P_bg46", "P_bg47", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v", "P_bg46", "P_bg47", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0", "P_bg46","P_bg47", bias=True),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj", "P_bg48","P_bg49", bias=True),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2", "P_bg50","P_bg51", bias=True),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q", "P_bg52", "P_bg53", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k", "P_bg54", "P_bg55", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v", "P_bg54", "P_bg55", bias=False),
     **dense("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0", "P_bg56","P_bg57", bias=True),
     **norm("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1", "P_bg57"),
     **norm("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2", "P_bg57"),
     **norm("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3", "P_bg57"),
     **conv("model.diffusion_model.input_blocks.4.1.proj_out", "P_bg57", "P_bg58"),   

     **easyblock("model.diffusion_model.input_blocks.5.0", "P_bg59", "P_bg60"),
     **norm("model.diffusion_model.input_blocks.5.1.norm", "P_bg61"),
     **conv("model.diffusion_model.input_blocks.5.1.proj_in", "P_bg61", "P_bg62"),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q", "P_bg63", "P_bg64", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k", "P_bg63", "P_bg64", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v", "P_bg63", "P_bg64", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0", "P_bg63","P_bg64", bias=True),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj", "P_bg65","P_bg66", bias=True),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2", "P_bg67","P_bg68", bias=True),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q", "P_bg69", "P_bg70", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k", "P_bg71", "P_bg72", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v", "P_bg71", "P_bg72", bias=False),
     **dense("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0", "P_bg73","P_bg74", bias=True),
     **norm("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1", "P_bg74"),
     **norm("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2", "P_bg74"),
     **norm("model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3", "P_bg74"),
     **conv("model.diffusion_model.input_blocks.5.1.proj_out", "P_bg74", "P_bg75"),

     **conv("model.diffusion_model.input_blocks.6.0.op", "P_bg76", "P_bg77"),
     **easyblock("model.diffusion_model.input_blocks.7.0", "P_bg78","P_bg79"),
     **conv("model.diffusion_model.input_blocks.7.0.skip_connection", "P_bg80","P_bg81"),
     **norm("model.diffusion_model.input_blocks.7.1.norm", "P_bg82"),
     **conv("model.diffusion_model.input_blocks.7.1.proj_in", "P_bg82", "P_bg83"),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q", "P_bg84", "P_bg85", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k", "P_bg84", "P_bg85", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v", "P_bg84", "P_bg85", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0", "P_bg84","P_bg85", bias=True),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj", "P_bg86","P_bg87", bias=True),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2", "P_bg88","P_bg89", bias=True),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q", "P_bg90", "P_bg91", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k", "P_bg92", "P_bg93", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v", "P_bg92", "P_bg93", bias=False),
     **dense("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0", "P_bg94","P_bg95", bias=True),
     **norm("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1", "P_bg95"),
     **norm("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2", "P_bg95"),
     **norm("model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3", "P_bg95"),
     **conv("model.diffusion_model.input_blocks.7.1.proj_out", "P_bg95", "P_bg96"),
     
     **easyblock("model.diffusion_model.input_blocks.8.0", "P_bg97","P_bg98"),
     **norm("model.diffusion_model.input_blocks.8.1.norm", "P_bg99"),
     **conv("model.diffusion_model.input_blocks.8.1.proj_in", "P_bg99", "P_bg100"),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q", "P_bg101", "P_bg102", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k", "P_bg101", "P_bg102", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v", "P_bg101", "P_bg102", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0", "P_bg101","P_bg102", bias=True),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj", "P_bg103","P_bg104", bias=True),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2", "P_bg105","P_bg106", bias=True),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q", "P_bg107", "P_bg108", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k", "P_bg109", "P_bg110", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v", "P_bg109", "P_bg110", bias=False),
     **dense("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0", "P_bg111","P_bg112", bias=True),
     **norm("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1", "P_bg112"),
     **norm("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2", "P_bg112"),
     **norm("model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3", "P_bg112"),
     **conv("model.diffusion_model.input_blocks.8.1.proj_out", "P_bg112", "P_bg113"),
     
     **conv("model.diffusion_model.input_blocks.9.0.op", "P_bg114", "P_bg115"),
     **easyblock("model.diffusion_model.input_blocks.10.0", "P_bg115", "P_bg116"),
     **easyblock("model.diffusion_model.input_blocks.11.0", "P_bg116", "P_bg117"),
     
     #middle blocks
     **easyblock("model.diffusion_model.middle_block.0", "P_bg117", "P_bg118"),
     **norm("model.diffusion_model.middle_block.1.norm", "P_bg119"),
     **conv("model.diffusion_model.middle_block.1.proj_in", "P_bg119", "P_bg120"),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q", "P_bg121", "P_bg122", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k", "P_bg121", "P_bg122", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v", "P_bg121", "P_bg122", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0", "P_bg121","P_bg122", bias=True),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj", "P_bg123","P_bg124", bias=True),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2", "P_bg125","P_bg126", bias=True),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q", "P_bg127", "P_bg128", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k", "P_bg129", "P_bg130", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v", "P_bg129", "P_bg130", bias=False),
     **dense("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0", "P_bg131","P_bg132", bias=True),
     **norm("model.diffusion_model.middle_block.1.transformer_blocks.0.norm1", "P_bg132"),
     **norm("model.diffusion_model.middle_block.1.transformer_blocks.0.norm2", "P_bg132"),
     **norm("model.diffusion_model.middle_block.1.transformer_blocks.0.norm3", "P_bg132"),
     **conv("model.diffusion_model.middle_block.1.proj_out", "P_bg132", "P_bg133"),
     
     **easyblock("model.diffusion_model.middle_block.2", "P_bg134", "P_bg135"),
       
     #output blocks
     **easyblock("model.diffusion_model.output_blocks.0.0", "P_bg136", "P_bg137"),
     **conv("model.diffusion_model.output_blocks.0.0.skip_connection","P_bg138","P_bg139"),

     **easyblock("model.diffusion_model.output_blocks.1.0", "P_bg140","P_bg141"),  
     **conv("model.diffusion_model.output_blocks.1.0.skip_connection","P_bg142","P_bg143"),

    
     **easyblock("model.diffusion_model.output_blocks.2.0", "P_bg144","P_bg145"),
     **conv("model.diffusion_model.output_blocks.2.0.skip_connection","P_bg146","P_bg147"),
     **conv("model.diffusion_model.output_blocks.2.1.conv", "P_bg148", "P_bg149"),
    
     **easyblock("model.diffusion_model.output_blocks.3.0", "P_bg150","P_bg151"),
     **conv("model.diffusion_model.output_blocks.3.0.skip_connection","P_bg152","P_bg153"),
     **norm("model.diffusion_model.output_blocks.3.1.norm", "P_bg154"),
     **conv("model.diffusion_model.output_blocks.3.1.proj_in", "P_bg154", "P_bg155"),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q", "P_bg156", "P_bg157", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k", "P_bg156", "P_bg157", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v", "P_bg156", "P_bg157", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0", "P_bg156","P_bg157", bias=True),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj", "P_bg158","P_bg159", bias=True),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2", "P_bg160","P_bg161", bias=True),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q", "P_bg162", "P_bg163", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k", "P_bg164", "P_bg165", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v", "P_bg164", "P_bg165", bias=False),
     **dense("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0", "P_bg166","P_bg167", bias=True),
     **norm("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1", "P_bg167"),
     **norm("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2", "P_bg167"),
     **norm("model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3", "P_bg167"),
     **conv("model.diffusion_model.output_blocks.3.1.proj_out", "P_bg167", "P_bg168"),

     **easyblock("model.diffusion_model.output_blocks.4.0", "P_bg169", "P_bg170"),
     **conv("model.diffusion_model.output_blocks.4.0.skip_connection","P_bg171","P_bg172"),
     **norm("model.diffusion_model.output_blocks.4.1.norm", "P_bg173"),
     **conv("model.diffusion_model.output_blocks.4.1.proj_in", "P_bg173", "P_bg174"),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q", "P_bg175", "P_bg176", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k", "P_bg175", "P_bg176", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v", "P_bg175", "P_bg176", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0", "P_bg175","P_bg176", bias=True),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj", "P_bg177","P_bg178", bias=True),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2", "P_bg179","P_bg180", bias=True),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q", "P_bg181", "P_bg182", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k", "P_bg183", "P_bg184", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v", "P_bg183", "P_bg184", bias=False),
     **dense("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0", "P_bg185","P_bg186", bias=True),
     **norm("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1", "P_bg186"),
     **norm("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2", "P_bg186"),
     **norm("model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3", "P_bg186"),
     **conv("model.diffusion_model.output_blocks.4.1.proj_out", "P_bg186", "P_bg187"),
     
     **easyblock("model.diffusion_model.output_blocks.5.0", "P_bg188", "P_bg189"),
     **conv("model.diffusion_model.output_blocks.5.0.skip_connection","P_bg190","P_bg191"),
     **norm("model.diffusion_model.output_blocks.5.1.norm", "P_bg192"),
     **conv("model.diffusion_model.output_blocks.5.1.proj_in", "P_bg192", "P_bg193"),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q", "P_bg194", "P_bg195", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k", "P_bg194", "P_bg195", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v", "P_bg194", "P_bg195", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0", "P_bg194","P_bg195", bias=True),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj", "P_bg196","P_bg197", bias=True),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2", "P_bg198","P_bg199", bias=True),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q", "P_bg200", "P_bg201", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k", "P_bg202", "P_bg203", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v", "P_bg202", "P_bg203", bias=False),
     **dense("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0", "P_bg204","P_bg205", bias=True),
     **norm("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1", "P_bg205"),
     **norm("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2", "P_bg205"),
     **norm("model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3", "P_bg205"),
     **conv("model.diffusion_model.output_blocks.5.1.proj_out", "P_bg205", "P_bg206"),
     **conv("model.diffusion_model.output_blocks.5.2.conv", "P_bg206", "P_bg207"),
     
     **easyblock("model.diffusion_model.output_blocks.6.0", "P_bg208","P_bg209"),
     **conv("model.diffusion_model.output_blocks.6.0.skip_connection","P_bg210","P_bg211"),
     **norm("model.diffusion_model.output_blocks.6.1.norm", "P_bg212"),
     **conv("model.diffusion_model.output_blocks.6.1.proj_in", "P_bg212", "P_bg213"),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q", "P_bg214", "P_bg215", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k", "P_bg214", "P_bg215", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v", "P_bg214", "P_bg215", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0", "P_bg214","P_bg215", bias=True),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj", "P_bg216","P_bg217", bias=True),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2", "P_bg218","P_bg219", bias=True),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q", "P_bg220", "P_bg221", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k", "P_bg222", "P_bg223", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v", "P_bg222", "P_bg223", bias=False),
     **dense("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0", "P_bg224","P_bg225", bias=True),
     **norm("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1", "P_bg225"),
     **norm("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2", "P_bg225"),
     **norm("model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3", "P_bg225"),
     **conv("model.diffusion_model.output_blocks.6.1.proj_out", "P_bg225", "P_bg226"),

     
     **easyblock("model.diffusion_model.output_blocks.7.0", "P_bg227", "P_bg228"),
     **conv("model.diffusion_model.output_blocks.7.0.skip_connection","P_bg229","P_bg230"),
     **norm("model.diffusion_model.output_blocks.7.1.norm", "P_bg231"),
     **conv("model.diffusion_model.output_blocks.7.1.proj_in", "P_bg231", "P_bg232"),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q", "P_bg233", "P_bg234", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k", "P_bg233", "P_bg234", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v", "P_bg233", "P_bg234", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0", "P_bg233","P_bg234", bias=True),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj", "P_bg235","P_bg236", bias=True),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2", "P_bg237","P_bg238", bias=True),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q", "P_bg239", "P_bg240", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k", "P_bg241", "P_bg242", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v", "P_bg241", "P_bg242", bias=False),
     **dense("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0", "P_bg243","P_bg244", bias=True),
     **norm("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1", "P_bg244"),
     **norm("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2", "P_bg244"),
     **norm("model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3", "P_bg244"),
     **conv("model.diffusion_model.output_blocks.7.1.proj_out", "P_bg244", "P_bg245"),

     **easyblock("model.diffusion_model.output_blocks.8.0", "P_bg246","P_bg247"),
     **conv("model.diffusion_model.output_blocks.8.0.skip_connection","P_bg248","P_bg249"),
     **norm("model.diffusion_model.output_blocks.8.1.norm", "P_bg250"),
     **conv("model.diffusion_model.output_blocks.8.1.proj_in", "P_bg250", "P_bg251"),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q", "P_bg252", "P_bg253", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k", "P_bg252", "P_bg253", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v", "P_bg252", "P_bg253", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0", "P_bg252","P_bg253", bias=True),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj", "P_bg254","P_bg255", bias=True),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2", "P_bg256","P_bg257", bias=True),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q", "P_bg258", "P_bg259", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k", "P_bg260", "P_bg261", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v", "P_bg260", "P_bg261", bias=False),
     **dense("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0", "P_bg262","P_bg263", bias=True),
     **norm("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1", "P_bg263"),
     **norm("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2", "P_bg263"),
     **norm("model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3", "P_bg263"),
     **conv("model.diffusion_model.output_blocks.8.1.proj_out", "P_bg263", "P_bg264"),     
     **conv("model.diffusion_model.output_blocks.8.2.conv", "P_bg265", "P_bg266"),
     
     **easyblock("model.diffusion_model.output_blocks.9.0", "P_bg267","P_bg268"),
     **conv("model.diffusion_model.output_blocks.9.0.skip_connection","P_bg269","P_bg270"), 
     **norm("model.diffusion_model.output_blocks.9.1.norm", "P_bg271"),
     **conv("model.diffusion_model.output_blocks.9.1.proj_in", "P_bg271", "P_bg272"),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q", "P_bg273", "P_bg274", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k", "P_bg273", "P_bg274", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v", "P_bg273", "P_bg274", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0", "P_bg273","P_bg274", bias=True),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj", "P_bg275","P_bg276", bias=True),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2", "P_bg277","P_bg278", bias=True),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q", "P_bg279", "P_bg280", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k", "P_bg281", "P_bg282", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v", "P_bg281", "P_bg282", bias=False),
     **dense("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0", "P_bg283","P_bg284", bias=True),
     **norm("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1", "P_bg284"),
     **norm("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2", "P_bg284"),
     **norm("model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3", "P_bg284"),
     **conv("model.diffusion_model.output_blocks.9.1.proj_out", "P_bg284", "P_bg285"),     

     **easyblock("model.diffusion_model.output_blocks.10.0", "P_bg286", "P_bg287"),
     **conv("model.diffusion_model.output_blocks.10.0.skip_connection","P_bg288","P_bg289"), 
     **norm("model.diffusion_model.output_blocks.10.1.norm", "P_bg290"),
     **conv("model.diffusion_model.output_blocks.10.1.proj_in", "P_bg290", "P_bg291"),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q", "P_bg292", "P_bg293", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k", "P_bg292", "P_bg293", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v", "P_bg292", "P_bg293", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0", "P_bg292","P_bg293", bias=True),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj", "P_b294","P_bg295", bias=True),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2", "P_bg296","P_bg297", bias=True),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q", "P_bg298", "P_bg299", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k", "P_bg300", "P_bg301", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v", "P_bg300", "P_bg301", bias=False),
     **dense("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0", "P_bg302","P_bg303", bias=True),
     **norm("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1", "P_bg303"),
     **norm("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2", "P_bg303"),
     **norm("model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3", "P_bg303"),
     **conv("model.diffusion_model.output_blocks.10.1.proj_out", "P_bg303", "P_bg304"),     

     **easyblock("model.diffusion_model.output_blocks.11.0", "P_bg305", "P_bg306"),
     **conv("model.diffusion_model.output_blocks.11.0.skip_connection","P_bg307","P_bg308"), 
     **norm("model.diffusion_model.output_blocks.11.1.norm", "P_bg309"),
     **conv("model.diffusion_model.output_blocks.11.1.proj_in", "P_bg309", "P_bg310"),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q", "P_bg311", "P_bg312", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k", "P_bg311", "P_bg312", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v", "P_bg311", "P_bg312", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0", "P_bg311","P_bg312", bias=True),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj", "P_bg313","P_bg314", bias=True),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2", "P_bg315","P_bg316", bias=True),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q", "P_bg317", "P_bg318", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k", "P_bg319", "P_bg320", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v", "P_bg319", "P_bg320", bias=False),
     **dense("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0", "P_bg321","P_bg322", bias=True),
     **norm("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1", "P_bg322"),
     **norm("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2", "P_bg322"),
     **norm("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3", "P_bg322"),
     **conv("model.diffusion_model.output_blocks.11.1.proj_out", "P_bg322", "P_bg323"),     

     **norm("model.diffusion_model.out.0", "P_bg324"),
     **conv("model.diffusion_model.out.2", "P_bg325", "P_bg326"),

     #Text Encoder
     #encoder down
     **conv("first_stage_model.encoder.conv_in", "P_bg327", "P_bg328"),  
     **easyblock2("first_stage_model.encoder.down.0.block.0", "P_bg328"),
     **easyblock2("first_stage_model.encoder.down.0.block.1", "P_bg328"),
     **conv("first_stage_model.encoder.down.0.downsample.conv", "P_bg328", "P_bg329"),
     
     **shortcutblock("first_stage_model.encoder.down.1.block.0", "P_bg330","P_bg331"),
     **easyblock2("first_stage_model.encoder.down.1.block.1", "P_bg331"),
     **conv("first_stage_model.encoder.down.1.downsample.conv", "P_bg331", "P_bg332"),
     
     **shortcutblock("first_stage_model.encoder.down.2.block.0", "P_bg332", "P_bg333"),
     **easyblock2("first_stage_model.encoder.down.2.block.1", "P_bg333"),
     **conv("first_stage_model.encoder.down.2.downsample.conv", "P_bg333", "P_bg334"),

     **easyblock2("first_stage_model.encoder.down.3.block.0", "P_bg334"),
     **easyblock2("first_stage_model.encoder.down.3.block.1", "P_bg334"),

     #encoder mid-block
     **easyblock2("first_stage_model.encoder.mid.block_1", "P_bg334"),

     **norm("first_stage_model.encoder.mid.attn_1.norm", "P_bg334"),
     **conv("first_stage_model.encoder.mid.attn_1.q", "P_bg334", "P_bg335"),
     **conv("first_stage_model.encoder.mid.attn_1.k", "P_bg334", "P_bg335"),
     **conv("first_stage_model.encoder.mid.attn_1.v", "P_bg334", "P_bg335"),
     **conv("first_stage_model.encoder.mid.attn_1.proj_out", "P_bg335", "P_bg336"),    

     **easyblock2("first_stage_model.encoder.mid.block_2", "P_bg336"),

     **norm("first_stage_model.encoder.norm_out", "P_bg337"),
     **conv("first_stage_model.encoder.conv_out", "P_bg338", "P_bg339"),

     **conv("first_stage_model.decoder.conv_in", "P_bg340", "P_bg341"),
     
     #decoder mid-block
     **easyblock2("first_stage_model.decoder.mid.block_1", "P_bg342"),
     **norm("first_stage_model.decoder.mid.attn_1.norm", "P_bg342"),
     **conv("first_stage_model.decoder.mid.attn_1.q", "P_bg342", "P_bg343"),
     **conv("first_stage_model.decoder.mid.attn_1.k", "P_bg342", "P_bg343"),
     **conv("first_stage_model.decoder.mid.attn_1.v", "P_bg342", "P_bg343"),
     **conv("first_stage_model.decoder.mid.attn_1.proj_out", "P_bg343", "P_bg344"),

     **easyblock2("first_stage_model.decoder.mid.block_2", "P_bg345"),
    
     #decoder up
     **shortcutblock("first_stage_model.decoder.up.0.block.0", "P_bg346","P_bg347"),
     **easyblock2("first_stage_model.decoder.up.0.block.1", "P_bg348"),
     **easyblock2("first_stage_model.decoder.up.0.block.2", "P_bg349"),

     **shortcutblock("first_stage_model.decoder.up.1.block.0", "P_bg350","P_bg351"),    
     **easyblock2("first_stage_model.decoder.up.1.block.1", "P_bg352"),
     **easyblock2("first_stage_model.decoder.up.1.block.2", "P_bg353"),
     **conv("first_stage_model.decoder.up.1.upsample.conv", "P_bg353", "P_bg354"),

     **easyblock2("first_stage_model.decoder.up.2.block.0", "P_bg355"),
     **easyblock2("first_stage_model.decoder.up.2.block.1", "P_bg355"),
     **easyblock2("first_stage_model.decoder.up.2.block.2", "P_bg355"),
     **conv("first_stage_model.decoder.up.2.upsample.conv", "P_bg355", "P_bg356"),

     **easyblock2("first_stage_model.decoder.up.3.block.0", "P_bg356"),
     **easyblock2("first_stage_model.decoder.up.3.block.1", "P_bg356"),
     **easyblock2("first_stage_model.decoder.up.3.block.2", "P_bg356"),
     **conv("first_stage_model.decoder.up.3.upsample.conv", "P_bg356", "P_bg357"),

     **norm("first_stage_model.decoder.norm_out", "P_bg358"),
     **conv("first_stage_model.decoder.conv_out", "P_bg359", "P_bg360"),
     **conv("first_stage_model.quant_conv", "P_bg361", "P_bg362"),
     **conv("first_stage_model.post_quant_conv", "P_bg363", "P_bg364"),

     **skip("cond_stage_model.transformer.text_model.embeddings.position_ids", None, None),

     **dense("cond_stage_model.transformer.text_model.embeddings.token_embedding","P_bg365", "P_bg366",bias=False),
     **dense("cond_stage_model.transformer.text_model.embeddings.position_embedding","P_bg367", "P_bg368",bias=False),

     #cond stage text encoder
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj", "P_bg369", "P_bg370",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj", "P_bg369", "P_bg370",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj", "P_bg369", "P_bg370",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj", "P_bg369", "P_bg370",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1", "P_bg370"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1", "P_bg370", "P_bg371", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2", "P_bg371", "P_bg372", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2", "P_bg372"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj", "P_bg372", "P_bg373",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj", "P_bg372", "P_bg373",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj", "P_bg372", "P_bg373",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj", "P_bg372", "P_bg373",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1", "P_bg373"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1", "P_bg373", "P_bg374", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2", "P_bg374", "P_bg375", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2", "P_bg375"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj", "P_bg375", "P_bg376",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj", "P_bg375", "P_bg376",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj", "P_bg375", "P_bg376",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj", "P_bg375", "P_bg376",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1", "P_bg376"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1", "P_bg376", "P_bg377", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2", "P_bg377", "P_bg378", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2", "P_bg378"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj", "P_bg378", "P_bg379",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj", "P_bg378", "P_bg379",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj", "P_bg378", "P_bg379",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj", "P_bg378", "P_bg379",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1", "P_bg379"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1", "P_bg379", "P_bg380", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2", "P_bg380", "P_b381", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2", "P_bg381"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj", "P_bg381", "P_bg382",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj", "P_bg381", "P_bg382",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj", "P_bg381", "P_bg382",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj", "P_bg381", "P_bg382",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1", "P_bg382"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1", "P_bg382", "P_bg383", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2", "P_bg383", "P_bg384", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2", "P_bg384"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj", "P_bg384", "P_bg385",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj", "P_bg384", "P_bg385",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj", "P_bg384", "P_bg385",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj", "P_bg384", "P_bg385",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1", "P_bg385"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1", "P_bg385", "P_bg386",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2", "P_bg386", "P_bg387",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2", "P_bg387"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj", "P_bg387", "P_bg388",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj", "P_bg387", "P_bg388",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj", "P_bg387", "P_bg388",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj", "P_bg387", "P_bg388",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1", "P_bg389"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1", "P_bg389", "P_bg390",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2", "P_bg390", "P_bg391", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2", "P_bg391"),
    
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj", "P_bg391", "P_bg392",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj", "P_bg391", "P_bg392",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj", "P_bg391", "P_bg392",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj", "P_bg391", "P_bg392",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1", "P_bg392"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1", "P_bg392", "P_bg393", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2", "P_bg393", "P_bg394", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2", "P_bg394"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj", "P_bg394", "P_bg395",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj", "P_bg394", "P_bg395",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj", "P_bg394", "P_bg395",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj", "P_bg394", "P_bg395",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1", "P_bg395"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1", "P_bg395", "P_bg396", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2", "P_bg396", "P_bg397", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2", "P_bg397"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj", "P_bg397", "P_bg398",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj", "P_bg397", "P_bg398",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj", "P_bg397", "P_bg398",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj", "P_bg397", "P_bg398",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1", "P_bg398"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1", "P_bg398", "P_bg399", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2", "P_bg400", "P_bg401", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2", "P_bg401"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj", "P_bg401", "P_bg402",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj", "P_bg401", "P_bg402",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj", "P_bg401", "P_bg402",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj", "P_bg401", "P_bg402",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1", "P_bg402"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1", "P_bg402", "P_bg403", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2", "P_bg403", "P_bg404", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2", "P_bg404"),

     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj", "P_bg404", "P_bg405",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj", "P_bg404", "P_bg405",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj", "P_bg404", "P_bg405",bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj", "P_bg404", "P_bg405",bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1", "P_bg405"),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1", "P_bg405", "P_bg406", bias=True),
     **dense("cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2", "P_bg406", "P_bg407", bias=True),
     **norm("cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2", "P_bg407"),
     
     **norm("cond_stage_model.transformer.text_model.final_layer_norm", "P_bg407")
    
      })



def cnn_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}

  return permutation_spec_from_axes_to_perm({
     **conv("conv1", None, "P_bg0"),
     **conv("conv2", "P_bg0", "P_bg1"),
     **dense("fc1", "P_bg1", "P_bg2"),
     **dense("fc2", "P_bg2", None, False),
  })




def resnet20_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

  # This is for easy blocks that use a residual connection, without any change in the number of channels.
  easyblock = lambda name, p: {
  **norm(f"{name}.bn1", p),
  **conv(f"{name}.conv1", p, f"P_{name}_inner"),
  **norm(f"{name}.bn2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p),
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
  **norm(f"{name}.bn1", p_in),
  **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
  **norm(f"{name}.bn2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
  **conv(f"{name}.shortcut.0", p_in, p_out),
  **norm(f"{name}.shortcut.1", p_out),
  }

  return permutation_spec_from_axes_to_perm({
    **conv("conv1", None, "P_bg0"),
    #
    **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
    **easyblock("layer1.1", "P_bg1",),
    **easyblock("layer1.2", "P_bg1"),
    #**easyblock("layer1.3", "P_bg1"),

    **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
    **easyblock("layer2.1", "P_bg2",),
    **easyblock("layer2.2", "P_bg2"),
    #**easyblock("layer2.3", "P_bg2"),

    **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
    **easyblock("layer3.1", "P_bg3",),
    **easyblock("layer3.2", "P_bg3"),
   # **easyblock("layer3.3", "P_bg3"),

    **norm("bn1", "P_bg3"),

    **dense("linear", "P_bg3", None),

})

# should be easy to generalize it to any depth
def resnet50_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

  # This is for easy blocks that use a residual connection, without any change in the number of channels.
  easyblock = lambda name, p: {
  **norm(f"{name}.bn1", p),
  **conv(f"{name}.conv1", p, f"P_{name}_inner"),
  **norm(f"{name}.bn2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p),
  }

  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
  shortcutblock = lambda name, p_in, p_out: {
  **norm(f"{name}.bn1", p_in),
  **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
  **norm(f"{name}.bn2", f"P_{name}_inner"),
  **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
  **conv(f"{name}.shortcut.0", p_in, p_out),
  **norm(f"{name}.shortcut.1", p_out),
  }

  return permutation_spec_from_axes_to_perm({
    **conv("conv1", None, "P_bg0"),
    #
    **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
    **easyblock("layer1.1", "P_bg1",),
    **easyblock("layer1.2", "P_bg1"),
    **easyblock("layer1.3", "P_bg1"),
    **easyblock("layer1.4", "P_bg1"),
    **easyblock("layer1.5", "P_bg1"),
    **easyblock("layer1.6", "P_bg1"),
    **easyblock("layer1.7", "P_bg1"),

    #**easyblock("layer1.3", "P_bg1"),

    **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
    **easyblock("layer2.1", "P_bg2",),
    **easyblock("layer2.2", "P_bg2"),
    **easyblock("layer2.3", "P_bg2"),
    **easyblock("layer2.4", "P_bg2"),
    **easyblock("layer2.5", "P_bg2"),
    **easyblock("layer2.6", "P_bg2"),
    **easyblock("layer2.7", "P_bg2"),

    **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
    **easyblock("layer3.1", "P_bg3",),
    **easyblock("layer3.2", "P_bg3"),
    **easyblock("layer3.3", "P_bg3"),
    **easyblock("layer3.4", "P_bg3"),
    **easyblock("layer3.5", "P_bg3"),
    **easyblock("layer3.6", "P_bg3"),
    **easyblock("layer3.7", "P_bg3"),

    **norm("bn1", "P_bg3"),

    **dense("linear", "P_bg3", None),

})



def vgg16_permutation_spec() -> PermutationSpec:
  layers_with_conv = [3,7,10,14,17,20,24,27,30,34,37,40]
  layers_with_conv_b4 = [0,3,7,10,14,17,20,24,27,30,34,37]
  layers_with_bn = [4,8,11,15,18,21,25,28,31,35,38,41]
  dense = lambda name, p_in, p_out, bias = True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
  return permutation_spec_from_axes_to_perm({
      # first features
      "features.0.weight": ( "P_Conv_0",None, None, None),
      "features.1.weight": ( "P_Conv_0", None),
      "features.1.bias": ( "P_Conv_0", None),
      "features.1.running_mean": ( "P_Conv_0", None),
      "features.1.running_var": ( "P_Conv_0", None),
      "features.1.num_batches_tracked": (),

      **{f"features.{layers_with_conv[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
        for i in range(len(layers_with_conv))},
      **{f"features.{i}.bias": (f"P_Conv_{i}", )
        for i in layers_with_conv + [0]},
      # bn
      **{f"features.{layers_with_bn[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", None)
        for i in range(len(layers_with_bn))},
      **{f"features.{layers_with_bn[i]}.bias": ( f"P_Conv_{layers_with_conv[i]}", None)
        for i in range(len(layers_with_bn))},
      **{f"features.{layers_with_bn[i]}.running_mean": ( f"P_Conv_{layers_with_conv[i]}", None)
        for i in range(len(layers_with_bn))},
      **{f"features.{layers_with_bn[i]}.running_var": ( f"P_Conv_{layers_with_conv[i]}", None)
        for i in range(len(layers_with_bn))},
      **{f"features.{layers_with_bn[i]}.num_batches_tracked": ()
        for i in range(len(layers_with_bn))},

      **dense("classifier", "P_Conv_40", "P_Dense_0", False),
})

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None, device=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      if device is not None:
        w = torch.index_select(w, axis, perm[p].int().to(device))
      else:
        w = torch.index_select(w, axis, perm[p].int())

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(ps: PermutationSpec, params_a, params_b, device, max_iter=100, init_perm=None):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    for p_ix in torch.randperm(len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = torch.zeros((n, n), device=device)
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk]                   
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis, device=device)
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
       
        A += w_a @ w_b.T

      ri, ci = linear_sum_assignment(A.cpu().detach().numpy(), maximize=True)
      assert (torch.tensor(ri, device=device) == torch.arange(len(ri), device=device)).all()
      
      oldL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n,device=device)[perm[p].long()]))
      newL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n,device=device)[ci, :]))
      print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = torch.Tensor(ci)

    if not progress:
      break

  return perm

def test_weight_matching():
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=3)
  print(ps.axes_to_perm)
  rng = torch.Generator()
  rng.manual_seed(13)
  num_hidden = 10
  shapes = {
      "layer0.weight": (2, num_hidden),
      "layer0.bias": (num_hidden, ),
      "layer1.weight": (num_hidden, 3),
      "layer1.bias": (3, )
  }

  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
  perm = weight_matching(rng, ps, params_a, params_b)
  print(perm)

if __name__ == "__main__":
  test_weight_matching()
