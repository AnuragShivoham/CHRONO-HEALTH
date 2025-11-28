#!/usr/bin/env python3
"""
convert_booster_to_js.py

Usage:
  python convert_booster_to_js.py /path/to/xgb_25k_fixed_booster.json /path/to/output/xgb_25k_inlined.js

This script converts an XGBoost JSON booster (gbtree) into an inlined JS implementation that:
- exposes `export function predict(f)` (f = feature array)
- each tree is a function tree_N(f) returning a scalar
- predict sums trees per target class using tree_info (XGBoost interleaves trees for multiclass)
- uses softmax to return probabilities

Notes:
- The script expects the booster JSON to contain:
  data['learner']['gradient_booster']['model']['trees']  (list of tree dicts)
  and data['learner']['learner_model_param']['num_class'] (num_class)
  and possibly data['learner']['gradient_booster']['model']['tree_info']
- It uses node arrays inside each tree:
  left_children, right_children, split_indices, split_conditions, base_weights.
- If JSON structure differs slightly, the script attempts a few common alternative paths.
"""

import json
import sys
import os

def find_booster_trees(data):
    # common paths
    try:
        return data['learner']['gradient_booster']['model']['trees']
    except Exception:
        pass
    try:
        return data['learner']['gradient_booster']['model']['gbtree']['trees']
    except Exception:
        pass
    # fallback: search for dict that looks like a tree list
    def find_trees(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'trees' and isinstance(v, list):
                    return v
                elif isinstance(v, (dict, list)):
                    res = find_trees(v)
                    if res:
                        return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_trees(item)
                if res:
                    return res
        return None
    return find_trees(data)

def get_num_class(data):
    try:
        return int(data['learner']['learner_model_param'].get('num_class', 1))
    except Exception:
        # fallback: try objective
        try:
            return int(data['learner']['objective']['softmax_multiclass_param']['num_class'])
        except Exception:
            return 1

def emit_tree_js(tree, tree_idx):
    """
    tree: dict with keys: left_children, right_children, split_indices, split_conditions, base_weights
    We'll create nested ifs. We'll assume left_children[i] == -1 indicates leaf.
    For leaf node, we'll find the corresponding leaf weight in base_weights:
      - If base_weights length == num_nodes: assume weight per node index
      - Else if base_weights length == 1: use base_weights[0]
      - Else if base_weights length > 1 and equals num_class: choose appropriate handling (return base_weights[0])
    """
    left = tree.get('left_children') or tree.get('left_children', [])
    right = tree.get('right_children') or tree.get('right_children', [])
    split_idx = tree.get('split_indices') or tree.get('split_indices', [])
    split_cond = tree.get('split_conditions') or tree.get('split_conditions', [])
    base_weights = tree.get('base_weights') or tree.get('base_weights', [])
    default_left = tree.get('default_left') or tree.get('default_left', [])
    # Some JSONs store children as ints; ensure lists of ints
    n_nodes = len(left)
    # Determine mapping of node index -> leaf weight
    # If base_weights length equals n_nodes, map directly node -> base_weights[node]
    leaf_weights_map = {}
    if isinstance(base_weights, list) and len(base_weights) == n_nodes:
        for i in range(n_nodes):
            leaf_weights_map[i] = base_weights[i]
    else:
        # Many boosters store base_weights as leaf values only for leaves.
        # We'll attempt to detect leaves and assign sequential leaf weights if base_weights length equals number of leaves.
        leaves = [i for i in range(n_nodes) if (left[i] < 0 and right[i] < 0)]
        if isinstance(base_weights, list) and len(base_weights) == len(leaves):
            for idx, node in enumerate(leaves):
                leaf_weights_map[node] = base_weights[idx]
        else:
            # fallback: if base_weights is scalar or list of one, use base_weights[0] for all leaves
            val = base_weights[0] if isinstance(base_weights, list) and len(base_weights) > 0 else (base_weights if isinstance(base_weights, (int,float)) else 0.0)
            for i in range(n_nodes):
                if left[i] < 0 and right[i] < 0:
                    leaf_weights_map[i] = val

    # Now recursively emit code
    sys.setrecursionlimit(10000)
    lines = []
    def emit_node(i, indent):
        sp = "  " * indent
        if left[i] < 0 and right[i] < 0:
            # leaf
            val = leaf_weights_map.get(i, 0.0)
            # ensure number literal JS-friendly
            if isinstance(val, float) or isinstance(val, int):
                lines.append(f"{sp}return {repr(float(val))};")
            else:
                lines.append(f"{sp}return {repr(float(val))};")
            return
        # non-leaf
        feat = split_idx[i] if i < len(split_idx) else 0
        cond = split_cond[i] if i < len(split_cond) else 0
        # handle NaN default: use default_left flag
        dl = default_left[i] if i < len(default_left) else 0
        # create condition: if (f[feat] <= cond) { left } else { right }
        # But if cond is very small/float, keep repr
        lines.append(f"{sp}if (f[{feat}] <= {repr(float(cond))}) {{")
        emit_node(left[i], indent+1)
        lines.append(f"{sp}}} else {{")
        emit_node(right[i], indent+1)
        lines.append(f"{sp}}}")
    # start function
    lines.append(f"function tree_{tree_idx}(f) {{")
    emit_node(0, 1)
    lines.append("}")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_booster_to_js.py booster.json out.js")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trees = find_booster_trees(data)
    if not trees:
        print("ERROR: Could not locate trees in JSON. Please check JSON structure.")
        sys.exit(1)

    num_class = get_num_class(data)
    tree_info = []
    # sometimes tree_info sits next to trees
    try:
        tree_info = data['learner']['gradient_booster']['model'].get('tree_info', [])
    except Exception:
        tree_info = []

    # If tree_info empty, populate with zeros (assume single class)
    if not tree_info:
        tree_info = [0]*len(trees)

    print(f"Found {len(trees)} trees, num_class={num_class}, tree_info len={len(tree_info)}")

    # Create JS file
    header = """// AUTO-GENERATED from booster JSON by convert_booster_to_js.py
// DO NOT EDIT MANUALLY
"""
    with open(outfile, 'w', encoding='utf-8') as out:
        out.write(header + "\n")
        # write tree functions
        for i, tree in enumerate(trees):
            js_tree = emit_tree_js(tree, i)
            out.write(js_tree + "\n\n")
        # write trees array and predict
        out.write("const trees = [\n")
        out.write(",\n".join([f"  tree_{i}" for i in range(len(trees))]))
        out.write("\n];\n\n")
        out.write("""function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const sum = exps.reduce((a,b) => a + b, 0);
  return exps.map(v => v / sum);
}

export function predict(f) {
  const numClasses = %d;
  let logits = new Array(numClasses).fill(0);
""" % (num_class))
        # sum trees according to tree_info: each tree belongs to tree_info[i] class
        # if tree_info mapping length matches trees length, use it; else assume interleaved by class
        if len(tree_info) == len(trees):
            # use explicit mapping
            for i in range(len(trees)):
                cls = int(tree_info[i])
                out.write(f"  logits[{cls}] += tree_{i}(f);\n")
        else:
            # assume interleaved: tree i goes to class = i % numClasses
            out.write("  for (let i=0;i<trees.length;i++){\n")
            out.write("    const cls = i % numClasses;\n")
            out.write("    logits[cls] += trees[i](f);\n")
            out.write("  }\n")
        out.write("""
  return softmax(logits);
}
""")
    print("Wrote", outfile)

if __name__ == "__main__":
    main()
