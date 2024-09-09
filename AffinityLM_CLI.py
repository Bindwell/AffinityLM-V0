import argparse
import json
import csv
import os
from models import AffinityLM  # Assuming the AffinityLM model is used here
import warnings
warnings.filterwarnings("ignore")

def write_json(smiles, pKds, filename):
    results = [{"SMILES": sm, "pKd": pkd} for sm, pkd in zip(smiles, pKds)]
    with open(filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def write_csv(smiles, pKds, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["SMILES", "pKd"])
        for sm, pkd in zip(smiles, pKds):
            writer.writerow([sm, pkd])

def determine_format_and_update_filename(output_arg, format_arg):
    if output_arg:
        _, ext = os.path.splitext(output_arg)
        if ext not in [".csv", ".json"]:
            output_arg += f".{format_arg or 'json'}"
        return output_arg, (format_arg or "json" if not ext else ext[1:])
    return None, "json"

def main():
    parser = argparse.ArgumentParser(description="Predict affinity using AffinityLM.")
    parser.add_argument("-t", "--target", nargs="+", required=True, help="The target protein sequence")
    parser.add_argument("-m", "--smiles", nargs="+", required=True, help="List of SMILES strings")
    parser.add_argument("-o", "--output", help="Optional output file path")
    parser.add_argument("-f", "--format", choices=["json", "csv"], help="Optional output file format; required if output is specified without an extension")
    parser.add_argument("-d", "--device", default="cpu", help="Specify the device for computation (default: cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for predictions")
    parser.add_argument("--protein_batch_size", type=int, default=16, help="Batch size for protein processing")
    parser.add_argument("--molecule_batch_size", type=int, default=16, help="Batch size for molecule processing")
    parser.add_argument("--save_cache", action="store_true", help="Option to save cache for faster future predictions")

    args = parser.parse_args()

    # Replace Plapt with AffinityLM (from app.py)
    affinity_model = AffinityLM(device=args.device)
    
    # Call the correct method: score_molecules
    results = affinity_model.score_molecules(
        protein=args.target[0], 
        molecules=args.smiles, 
        batch_size=args.batch_size,
        prot_batch_size=args.protein_batch_size,
        mol_batch_size=args.molecule_batch_size,
        save_cache=args.save_cache
    )

    # Extract SMILES and pKd from the results
    results.to_csv('')
if __name__ == "__main__":
    main()
