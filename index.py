from flask import Flask, request, jsonify
import warnings
from models import AffinityLM  # Assuming the AffinityLM model is used here

warnings.filterwarnings("ignore")

app = Flask(__name__)

affinity_model = AffinityLM(device="cpu")

@app.route('/', methods=['POST'])
def score_molecules():
    data = request.get_json()
    
    target_protein = data.get("target", "")
    smiles_list = data.get("smiles", [])
    print(target_protein, smiles_list)
    if not target_protein or not smiles_list:
        return jsonify({"error": "Please provide a valid target and list of SMILES strings."}), 400
    results = affinity_model.score_molecules(
        protein=target_protein, 
        molecules=smiles_list
    )
    print(results)
    results_dict = results.to_dict(orient="records")
    return jsonify(results_dict), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
