from collections import defaultdict
from re import L
from typing import NamedTuple
import torch
from scipy.optimize import linear_sum_assignment
import time
from random import shuffle

rngmix = lambda rng, x: random.fold_in(rng, hash(x))

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

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  
  try:
    for axis, p in enumerate(ps.axes_to_perm[k]):
      # Skip the axis we're trying to permute.
      if axis == except_axis:
        continue

      # None indicates that there is no permutation relevant to that axis.
      if p is not None:
        w = torch.index_select(w, axis, perm[p].int())
  except:   
    #print("error in layer {}".format(k))
    #rint("")
    print(k)
    #print(ps.axes_to_perm.keys())
    raise Exception("error")
    #print(axis)
    #print(p)
    #print(perm[p].int())
    #raise Exception("RuntimeError: mat1 and mat2 shapes cannot be multiplied (5120x960 and 320x5120)")

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys() if "model_" not in k}

def weight_matching(ps: PermutationSpec, params_a, params_b, special_layers=None, device="cpu", max_iter=3, init_perm=None, usefp16=False):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items() if axes[0][0] in params_b}
  #print(perm_sizes)
  perm = dict()
  perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  special_layers = special_layers if special_layers and len(special_layers) > 0 else sorted(list(perm.keys()))
  #print(special_layers)
  sum = 0
  number = 0

  if usefp16:
    for iteration in range(max_iter):
      progress = False
      shuffle(special_layers)
      for p_ix in special_layers:
        p = p_ix
        if p in special_layers:
          n = perm_sizes[p]
          A = torch.zeros((n, n), dtype=torch.float16).to(device)
          for wk, axis in ps.perm_to_axes[p]:
              w_a = params_a[wk]
              w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
              w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
              w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)
              A += torch.matmul(w_a.half(), w_b.half())

          A = A.cpu()
          ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)

          assert (torch.tensor(ri) == torch.arange(len(ri))).all()
          
          oldL = torch.vdot(torch.flatten(A).float(), torch.flatten(torch.eye(n)[perm[p].long()]).float()).half()
          newL = torch.vdot(torch.flatten(A).float(), torch.flatten(torch.eye(n)[ci, :]).float()).half()
          
          if newL - oldL != 0:
            sum += abs((newL-oldL).item())
            number += 1
            print(f"{p}: {newL - oldL}")

          progress = progress or newL > oldL + 1e-12

          perm[p] = torch.Tensor(ci)
        
      if not progress:
        break
    
    if number > 0:
      average = sum / number
    else:
      average = 0
    return (perm, average)

  else:
    for iteration in range(max_iter):
      progress = False
      shuffle(special_layers)
      for p_ix in special_layers:
        p = p_ix
        if p in special_layers:
          n = perm_sizes[p]
          A = torch.zeros((n, n), dtype=torch.float32).to(device="cpu")
          for wk, axis in ps.perm_to_axes[p]:
            w_a = params_a[wk]
            w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
            w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
            w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)
            A += torch.matmul(w_a.float(), w_b.float()).cpu()

          ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)

          assert (torch.tensor(ri) == torch.arange(len(ri))).all()
        
          oldL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n)[perm[p].long()]).float())
          newL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n)[ci, :]).float())

          if newL - oldL != 0:
            sum += abs((newL-oldL).item())
            number += 1
            print(f"{p}: {newL - oldL}")

          progress = progress or newL > oldL + 1e-12

          perm[p] = torch.Tensor(ci)
        
      if not progress:
        break

    if number > 0:
      average = sum / number
    else:
      average = 0
    return (perm, average)


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
