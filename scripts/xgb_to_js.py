import json
import os

MODEL_PATH = "Data/processed/xgb_25k_model.json"
OUTPUT_PATH = "Data/processed/xgb_25k_model.js"

print("📥 Loading model:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: JSON model not found!")
    exit()

with open(MODEL_PATH, "r") as f:
    data = json.load(f)

if "trees" not in data:
    print("❌ ERROR: Invalid XGBoost JSON format. No 'trees' key found.")
    exit()

print(f"🌲 Total trees: {len(data['trees'])}")

js = "// Auto-generated JS from XGBoost JSON\n\n"

# ---- TREE GENERATOR ----
def generate_node(node, nodes, depth=1):
    indent = "    " * depth

    # Leaf node
    if "leaf" in node:
        return f"{indent}return {node['leaf']};\n"

    # Non-leaf node
    fid = node["split"]
    thresh = node["split_condition"]
    yes = node["yes"]
    no = node["no"]

    code = f"{indent}if (features[{fid}] <= {thresh}) {{\n"
    code += generate_node(nodes[yes], nodes, depth + 1)
    code += f"{indent}}} else {{\n"
    code += generate_node(nodes[no], nodes, depth + 1)
    code += f"{indent}}}\n"

    return code


# ---- BUILD TREE FUNCTIONS ----
for idx, tree in enumerate(data["trees"]):
    js += f"// Tree {idx}\n"
    js += f"function tree_{idx}(features) {{\n"
    js += generate_node(tree["nodes"][0], tree["nodes"], 1)
    js += "}\n\n"

# ---- FINAL PREDICT FUNCTION ----
js += "export function predict(features) {\n"
js += "    let sum = 0;\n"

for idx in range(len(data["trees"])):
    js += f"    sum += tree_{idx}(features);\n"

js += "    return sum;\n"
js += "}\n"

# ---- SAVE JS FILE ----
with open(OUTPUT_PATH, "w") as f:
    f.write(js)

print("✅ JS model generated successfully!")
print("➡ Output:", OUTPUT_PATH)
