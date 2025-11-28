import json
import os

INPUT_JSON = "Data/processed/xgb_25k_fixed_booster.json"
OUTPUT_JS = "Data/processed/xgb_25k_inlined.js"

def build_js_node(node):
    """Recursively convert any XGBoost node into JS."""

    # Case 1: Pure leaf
    if "leaf" in node:
        return f"return {node['leaf']};"

    # Case 2: No split = leaf (fallback)
    if "split" not in node:
        return f"return {node.get('leaf', 0)};"

    fid = node["split"]
    threshold = node["split_condition"]

    children = node.get("children", [])

    # If no children, treat as leaf
    if not children:
        return f"return {node.get('leaf', 0)};"

    # Find yes/no nodes
    yes = next((c for c in children if c["nodeid"] == node["yes"]), None)
    no = next((c for c in children if c["nodeid"] == node["no"]), None)

    # Fallback if missing
    if yes is None or no is None:
        return f"return {node.get('leaf', 0)};"

    return (
        f"if (f['{fid}'] <= {threshold}) {{ {build_js_node(yes)} }} "
        f"else {{ {build_js_node(no)} }}"
    )


def build_js_tree(tree, idx):
    """Each tree object is the root node."""
    root = tree
    body = build_js_node(root)
    return f"""
function tree_{idx}(f) {{
    {body}
}}
"""


def build_js_model(model, num_classes):
    trees = model["trees"]

    js = ["// AUTO-GENERATED INLINE XGB MODEL"]

    for i, tree in enumerate(trees):
        js.append(build_js_tree(tree, i))

    js.append("""
function softmax(arr) {
    const m = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - m));
    const sum = exps.reduce((a,b) => a+b, 0);
    return exps.map(v => v / sum);
}
""")

    js.append(f"""
export function predict(f) {{
    let logits = new Array({num_classes}).fill(0);

    for (let i = 0; i < {len(trees)}; i++) {{
        const cls = i % {num_classes};
        logits[cls] += tree_{i}(f);
    }}

    return softmax(logits);
}}
""")

    return "\n".join(js)


### MAIN EXECUTION ###
print("📥 Loading booster JSON...")
data = json.load(open(INPUT_JSON, "r", encoding="utf-8"))

learner = data["learner"]
model = learner["gradient_booster"]["model"]
num_classes = int(learner["learner_model_param"]["num_class"])

print("🔄 Converting booster → JavaScript...")
js_code = build_js_model(model, num_classes)

open(OUTPUT_JS, "w", encoding="utf-8").write(js_code)

print("🎉 DONE! JS file created:")
print("➡", OUTPUT_JS)
