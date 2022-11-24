import argparse
import torch
import os
import copy

from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation


parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("--model_a", type=str, help="Path to model a")
parser.add_argument("--model_b", type=str, help="Path to model b")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", type=str, help="Whether to use half precision", default=True, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)
args = parser.parse_args()   
device = args.device

def flatten_params(model):
  return model["state_dict"]

model_a = torch.load(args.model_a, map_location=device)
model_b = torch.load(args.model_b, map_location=device)
theta_0 = model_a["state_dict"]
theta_1 = model_b["state_dict"]

alpha = float(args.alpha)
iterations = int(args.iterations)
step = alpha/iterations

if args.usefp16:
    print("Using half precision")
else:
    print("Using full precision")

permutation_spec = sdunet_permutation_spec()

for x in range(iterations):
    print(f"""
    ---------------------
         ITERATION {x+1}
    ---------------------
    """)

    # In order to reach a certain alpha value with a given number of steps,
    # You have to calculate an alpha for each individual iteration
    if x > 0:
        new_alpha = 1 - (1 - step*(1+x)) / (1 - step*(x))
    else:
        new_alpha = step
    print(f"new alpha = {new_alpha}\n")

    # Add the models together in specific ratio to reach final ratio
    for key in theta_0.keys():
        if "model" in key and key in theta_1:
            theta_0[key] = (1 - (new_alpha)) * (theta_0[key]) + (new_alpha) * (theta_1[key])

    if x == 0:
        for key in theta_1.keys():
            if "model" in key and key not in theta_0:
                theta_0[key] = theta_1[key]

    print("FINDING PERMUTATIONS")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, flatten_params(model_a), theta_0, usefp16=args.usefp16)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(permutation_spec, flatten_params(model_b), theta_0, usefp16=args.usefp16)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

    new_alpha = y / (y + z)
    print(new_alpha)

    # Weighted sum of the permutations
    for key in theta_0.keys():
        if "model" in key and key in theta_1:
            theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])

output_file = f'{args.output}.ckpt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

print("\nSaving...")

torch.save({
        "state_dict": theta_0
            }, output_file)

print("Done!")
