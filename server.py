from flask import Flask, request, jsonify, render_template
from calculate_truss import analyze_truss  # Assuming analyze_truss handles numeric inputs
import numpy as np  # Import numpy

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    force_angle = data.get("force_angle")
    roller_angle = data.get("roller_angle")
    force_magnitude = data.get("force_magnitude")

    if force_angle is None or roller_angle is None:
        return jsonify({"error": "Both force_angle and roller_angle are required."}), 400
    
    if not isinstance(force_angle, (float, int)) or not isinstance(roller_angle, (float, int)):
        return jsonify({"error": "Invalid input types. Only numeric values are allowed."}), 400

    U, R, S, nodes, deformed_nodes = analyze_truss(force_angle=force_angle, roller_angle=roller_angle, force_magnitude=force_magnitude)

    nodal_displacements = {
        node: U[2*i:2*i+2]
        for i, node in enumerate(['A', 'B', 'C', 'D'])
    }

    for key, value in nodal_displacements.items():
        nodal_displacements[key] = [round(val, 6) for val in value.tolist()]

    for key, value in nodes.items():
        nodes[key] = [round(val, 6) for val in value.tolist()]

    for key, value in deformed_nodes.items():
        deformed_nodes[key] = [round(val, 6) for val in value.tolist()]

    for key, value in R.items():
        R[key] = [round(val, 6) for val in value.tolist()]
    

    results = {
        "nodal_displacements": nodal_displacements,
        "reaction_forces": {
            node: [round(force_value, 6) for force_value in force] 
            for node, force in R.items()
        },
        "element_stresses": {
            element: round(stress, 6)
            for element, stress in S.items()
        },
        "original_nodes": nodes, 
        "deformed_nodes": deformed_nodes ,
        "elements": {
            1: ['A', 'B'],
            2: ['A', 'C'],
            3: ['A', 'D'],
            4: ['B', 'C'],
            5: ['C', 'D']
        }
    }

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)