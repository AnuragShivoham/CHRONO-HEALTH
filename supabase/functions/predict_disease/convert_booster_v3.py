#!/usr/bin/env python3
"""
convert_booster_v3.py

Robust converter for XGBoost JSON boosters:
- supports flat-array tree format (left_children/right_children/split_indices/...)
- supports nested node format (nodeid/children/yes/no/missing/leaf)
- outputs Deno-safe inline JS:
    - function tree_i(f) { ... }
    - const trees = [ tree_0, tree_1, ... ];
    - export function predict(f) { ... }
Usage:
  python convert_booster_v3.py Data/processed/xgb_25k_fixed_booster.json supabase/functions/predict_disease/xgb_25k_inlined.js
"""

import json
import sys
import math

def find_booster_section(data):
    # common paths
    for path in [
        ('learner','gradient_booster','model'),
        ('learner','gradient_booster'),
        ('learner','gradient_booster','model','gbtree'),
        ()
    ]:
        cur = data
        ok = True
        if not path:
            continue
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, dict):
            return cur
    # fallback: search recursively for object that has 'trees' list
    def search(o):
        if isinstance(o, dict):
            if 'trees' in o and isinstance(o['trees'], list):
                return o
            for v in o.values():
                r = search(v)
                if r is not None: return r
        elif isinstance(o, list):
            for item in o:
                r = search(item)
                if r is not None: return r
        return None
    return search(data)

# ------------------ Emit JS for flat-array tree ------------------
def emit_flat_tree_js(tree, idx):
    # tree expected keys: left_children, right_children, split_indices, split_conditions, base_weights (leaf values), default_left
    left = tree.get('left_children') or tree.get('left_children', [])
    right = tree.get('right_children') or tree.get('right_children', [])
    splits = tree.get('split_indices') or tree.get('split_indices', [])
    conds = tree.get('split_conditions') or tree.get('split_conditions', [])
    base_weights = tree.get('base_weights') or tree.get('base_weights', [])
    default_left = tree.get('default_left') or tree.get('default_left', [])

    n = max(len(left), len(right), len(splits), len(conds))
    # fallback: if base_weights length equals n, treat as node-wise weight; otherwise leaves
    # build map of node -> leaf value if possible
    leaf_map = {}
    if isinstance(base_weights, list) and len(base_weights) == n:
        for i in range(n):
            leaf_map[i] = float(base_weights[i])
    else:
        # detect leaves by left[i] < 0 and right[i] < 0
        leaves = [i for i in range(n) if (i < len(left) and i < len(right) and left[i] < 0 and right[i] < 0)]
        if isinstance(base_weights, list) and len(base_weights) == len(leaves) and len(leaves) > 0:
            for j,node in enumerate(leaves):
                leaf_map[node] = float(base_weights[j])
        else:
            # fallback zero for leaves not found
            for i in leaves:
                leaf_map[i] = 0.0

    lines = []
    lines.append(f"function tree_{idx}(f) {{")
    sys.setrecursionlimit(10000)

    def emit_node(i, indent):
        sp = "  " * indent
        # protective bounds
        li = left[i] if i < len(left) else -1
        ri = right[i] if i < len(right) else -1
        if li < 0 and ri < 0:
            val = leaf_map.get(i, 0.0)
            lines.append(f"{sp}return {float(val)};")
            return
        feat = splits[i] if i < len(splits) else 0
        cond = conds[i] if i < len(conds) else 0.0
        # default_left fallback
        dl = default_left[i] if i < len(default_left) else 0
        # emit safe condition
        lines.append(f"{sp}if (f[{int(feat)}] <= {float(cond)}) "+"{")
        # left branch
        if li >= 0:
            emit_node(int(li), indent+1)
        else:
            # try to use leaf_map for this index if any
            val = leaf_map.get(i, 0.0)
            lines.append(f"{sp}  return {float(val)};")
        lines.append(f"{sp}"+"} else {")
        # right branch
        if ri >= 0:
            emit_node(int(ri), indent+1)
        else:
            val = leaf_map.get(i, 0.0)
            lines.append(f"{sp}  return {float(val)};")
        lines.append(f"{sp}"+"}")

    # If no left/right arrays, fallback to stub
    if not left or not right:
        lines.append("  return 0.0;")
    else:
        emit_node(0, 1)
    lines.append("}")
    return "\n".join(lines)

# ------------------ Emit JS for nested-node tree ------------------
def emit_nested_tree_js(root, idx):
    # root is a dict with nodeid, split, split_condition, yes/no/missing, children OR nested nodes
    lines = []
    lines.append(f"function tree_{idx}(f) {{")
    sys.setrecursionlimit(10000)

    # build map nodeid->node by traversing nested structure
    node_map = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        nid = n.get('nodeid')
        if nid is not None:
            node_map[int(nid)] = n
        ch = n.get('children') or []
        if isinstance(ch, list):
            for c in ch:
                stack.append(c)
        elif isinstance(ch, dict):
            stack.append(ch)
        # also check nested keys
        for k in ['left','right']:
            if k in n and isinstance(n[k], dict):
                stack.append(n[k])

    def emit_node(nid, indent):
        sp = "  " * indent
        node = node_map.get(nid)
        if node is None:
            lines.append(f"{sp}return 0.0;")
            return
        if 'leaf' in node:
            lines.append(f"{sp}return {float(node.get('leaf',0.0))};")
            return
        # determine feature index
        feat = node.get('split_index') or node.get('split_feature') or node.get('split')
        try:
            feat_idx = int(feat)
        except Exception:
            # fallback 0
            feat_idx = 0
        cond = node.get('split_condition') if 'split_condition' in node else node.get('threshold',0.0)
        yes = node.get('yes')
        no = node.get('no')
        missing = node.get('missing', yes)
        # if no/yes missing, try children order
        if (yes is None or no is None) and 'children' in node:
            ch = node.get('children') or []
            if isinstance(ch, list) and len(ch) >= 2:
                yes = ch[0].get('nodeid')
                no = ch[1].get('nodeid')
        yes = yes if yes is not None else -1
        no = no if no is not None else -1

        lines.append(f"{sp}if (f[{feat_idx}] <= {float(cond)}) "+"{")
        if yes is not None and int(yes) in node_map:
            emit_node(int(yes), indent+1)
        else:
            lines.append(f"{sp}  return 0.0;")
        lines.append(f"{sp}"+"} else {")
        if no is not None and int(no) in node_map:
            emit_node(int(no), indent+1)
        else:
            lines.append(f"{sp}  return 0.0;")
        lines.append(f"{sp}"+"}")

    # pick root id as 0 if present else min nodeid
    root_id = 0
    if 0 not in node_map:
        root_id = min(node_map.keys()) if node_map else 0
    emit_node(root_id, 1)
    lines.append("}")
    return "\n".join(lines)

# ------------------ Main converter ------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_booster_v3.py booster.json out.js")
        sys.exit(1)

    booster_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(booster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    booster_section = find_booster_section(data)
    if booster_section is None:
        print("ERROR: could not find booster section in JSON")
        sys.exit(1)

    # locate trees list
    trees = booster_section.get('trees') or data.get('trees') or []
    if not isinstance(trees, list) or len(trees) == 0:
        # try deeper scanning
        def scan(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k == 'trees' and isinstance(v, list):
                        return v
                    else:
                        r = scan(v)
                        if r: return r
            elif isinstance(obj, list):
                for it in obj:
                    r = scan(it)
                    if r: return r
            return None
        trees = scan(data) or []
    if not trees:
        print("ERROR: no trees found")
        sys.exit(1)

    # find tree_info and num_class
    tree_info = booster_section.get('tree_info') or data.get('learner',{}).get('gradient_booster',{}).get('model',{}).get('tree_info') or []
    num_class = 1
    try:
        num_class = int(data.get('learner',{}).get('learner_model_param',{}).get('num_class',1))
    except Exception:
        num_class = 1

    print(f"Found {len(trees)} trees, num_class={num_class}, tree_info_len={len(tree_info)}")

    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("// AUTO-GENERATED INLINE XGBOOST JS MODEL (v3)\n\n")
        warnings = 0
        for i, tree in enumerate(trees):
            try:
                # detect flat-array structure
                if isinstance(tree, dict) and ('left_children' in tree or 'right_children' in tree) and isinstance(tree.get('left_children',None), list):
                    js = emit_flat_tree_js(tree, i)
                    out.write(js + "\n\n")
                # detect nested node format where root contains 'nodeid' or 'children'
                elif isinstance(tree, dict) and ('nodeid' in tree or 'children' in tree or 'leaf' in tree or 'split_condition' in tree):
                    js = emit_nested_tree_js(tree, i)
                    out.write(js + "\n\n")
                # detect 'nodes' list with node dicts
                elif isinstance(tree, dict) and 'nodes' in tree and isinstance(tree['nodes'], list):
                    # build node map from nodes and use emit_nested_tree_js
                    nodes = tree['nodes']
                    # build a pseudo-root that contains children list
                    root = None
                    # find node with nodeid == 0 or with no parent
                    node_map = {int(n['nodeid']): n for n in nodes if 'nodeid' in n}
                    root_id = 0 if 0 in node_map else min(node_map.keys())
                    # create fake nested structure by leaving nodes as-is and emit from map
                    js = emit_tree_from_map_for_nodes(node_map, root_id, i) if 'emit_tree_from_map_for_nodes' in globals() else emit_nested_tree_js(nodes[0], i)
                    out.write(js + "\n\n")
                else:
                    # fallback: try to build node map from nested search
                    # attempt to find first dict-with-nodeid in this tree
                    found = False
                    def find_nodeobj(o):
                        if isinstance(o, dict):
                            if 'nodeid' in o:
                                return o
                            for v in o.values():
                                r = find_nodeobj(v)
                                if r: return r
                        elif isinstance(o, list):
                            for it in o:
                                r = find_nodeobj(it)
                                if r: return r
                        return None
                    rootobj = find_nodeobj(tree)
                    if rootobj:
                        js = emit_nested_tree_js(rootobj, i)
                        out.write(js + "\n\n")
                    else:
                        warnings += 1
                        out.write(f"function tree_{i}(f) {{ return 0.0; }}\n\n")
                        if warnings <= 5:
                            print(f"Warning: cannot determine root for tree {i}, emitting stub.")
            except Exception as e:
                warnings += 1
                out.write(f"function tree_{i}(f) {{ return 0.0; }}\n\n")
                if warnings <= 5:
                    print(f"Warning generating tree {i}: {e}")
        # trees array
        out.write("const trees = [\n")
        out.write(",\n".join([f"  tree_{i}" for i in range(len(trees))]))
        out.write("\n];\n\n")
        # softmax + predict
        out.write("""
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(v => v / sum);
}

export function predict(f) {
  const numClasses = %d;
  let logits = new Array(numClasses).fill(0);
""" % num_class)
        if tree_info and len(tree_info) == len(trees):
            for i,cls in enumerate(tree_info):
                out.write(f"  logits[{int(cls)}] += tree_{i}(f);\n")
        else:
            out.write("  for (let i=0;i<trees.length;i++){\n")
            out.write("    const cls = i % numClasses;\n")
            out.write("    logits[cls] += trees[i](f);\n")
            out.write("  }\n")
        out.write("""
  return softmax(logits);
}
""")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
