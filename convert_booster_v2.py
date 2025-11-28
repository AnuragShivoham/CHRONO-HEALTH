#!/usr/bin/env python3
"""
convert_booster_v2.py

Converts an XGBoost JSON booster (nested node format) into a Deno-safe inline JS model:
- emits function tree_N(f) for each tree
- emits export function predict(f) returning softmax probabilities
- supports nested nodes with nodeid, yes/no/missing, children, leaf, split_condition, split_index

Usage:
  python convert_booster_v2.py /path/to/xgb_25k_fixed_booster.json /path/to/out/xgb_25k_inlined.js
"""

import json
import sys
import os

def find_booster(data):
    # try common locations
    try:
        return data['learner']['gradient_booster']['model']
    except Exception:
        pass
    try:
        return data['learner']['gradient_booster']
    except Exception:
        pass
    # fallback: search for dict that contains 'trees'
    def search(obj):
        if isinstance(obj, dict):
            if 'trees' in obj and isinstance(obj['trees'], list):
                return obj
            for v in obj.values():
                res = search(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = search(item)
                if res is not None:
                    return res
        return None
    return search(data)

def build_node_map(tree_root):
    """Traverse nested children to build nodeid -> node mapping."""
    node_map = {}
    stack = [tree_root]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        nid = n.get('nodeid')
        if nid is not None:
            node_map[int(nid)] = n
        # children can be under 'children' as list
        ch = n.get('children') or []
        if isinstance(ch, dict):
            stack.append(ch)
        elif isinstance(ch, list):
            for c in ch:
                stack.append(c)
        # Some formats put 'left'/'right' or nothing; also traverse yes/no references by scanning entire tree
    return node_map

def emit_tree_from_map(node_map, root_id, idx):
    """Emit JS function for given tree using node_map and yes/no links."""
    lines = []
    lines.append(f"function tree_{idx}(f) {{")

    def emit_node(node_id, indent):
        node = node_map.get(node_id)
        sp = "  " * indent
        if node is None:
            lines.append(f"{sp}return 0.0;")
            return

        # leaf case
        if 'leaf' in node:
            val = node.get('leaf', 0.0)
            # format numeric
            lines.append(f"{sp}return {float(val)};")
            return

        # get feature index
        feat_idx = None
        # common keys: 'split_index', 'split_feature', 'split'
        if 'split_index' in node:
            feat_idx = node.get('split_index')
        elif 'split_feature' in node:
            feat_idx = node.get('split_feature')
        elif 'split' in node:
            # sometimes split is feature *name* — try numeric parse fallback
            try:
                feat_idx = int(node.get('split'))
            except Exception:
                feat_idx = node.get('split')  # string feature name (we hope split_index exists somewhere else)
        # get condition
        cond = node.get('split_condition') if 'split_condition' in node else node.get('split_condition', None)
        if cond is None:
            # some JSON uses 'threshold'
            cond = node.get('threshold', 0.0)

        # get child ids (yes/no/missing)
        yes = node.get('yes')
        no = node.get('no')
        missing = node.get('missing')

        # If children array exists without yes/no, try to extract by nodeid in children
        if yes is None or no is None:
            ch = node.get('children') or []
            if isinstance(ch, list) and len(ch) >= 2:
                # try to use their nodeid fields; find which is yes/no by looking for their nodeid values present
                try:
                    yes = ch[0].get('nodeid')
                    no = ch[1].get('nodeid')
                except Exception:
                    pass

        # Safety defaults
        yes = yes if yes is not None else -1
        no = no if no is not None else -1
        # default missing branch: if not provided, use yes
        missing = missing if missing is not None else yes

        # If feat_idx is not integer (e.g., name), we cannot index — fallback to 0
        if isinstance(feat_idx, str):
            # try to convert numeric substring
            try:
                feat_idx = int(feat_idx)
            except Exception:
                feat_idx = 0

        # Emit condition
        # Use <= as in many XGBoost dumps
        lines.append(f"{sp}if (f[{feat_idx}] <= {float(cond)}) "+"{")
        if yes is not None and yes != -1:
            emit_node(int(yes), indent+1)
        else:
            lines.append(f"{sp}  return 0.0;")
        lines.append(f"{sp}"+"} else {")
        if no is not None and no != -1:
            emit_node(int(no), indent+1)
        else:
            lines.append(f"{sp}  return 0.0;")
        lines.append(f"{sp}"+"}")
    # Start at root_id
    emit_node(int(root_id), 1)
    lines.append("}")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_booster_v2.py booster.json out.js")
        sys.exit(1)

    booster_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(booster_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    booster = find_booster(data)
    if booster is None:
        print("ERROR: Could not locate booster/model structure in JSON.")
        sys.exit(1)

    # trees location
    trees = booster.get('trees') or booster.get('tree_info') and booster.get('trees') or None
    if trees is None:
        # try deeper
        trees = data.get('trees') or data.get('learner', {}).get('gradient_booster', {}).get('model', {}).get('trees')
    if not trees or not isinstance(trees, list):
        print("ERROR: No trees found at expected locations.")
        # try scanning top-level for 'trees'
        def scan_for_trees(obj):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if k == 'trees' and isinstance(v, list):
                        return v
                    else:
                        res = scan_for_trees(v)
                        if res: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = scan_for_trees(item)
                    if res: return res
            return None
        trees = scan_for_trees(data)
        if not trees:
            print("ERROR: still could not find trees")
            sys.exit(1)

    # Get tree_info (class mapping) and num_class
    tree_info = booster.get('tree_info') or []
    try:
        num_class = int(data['learner']['learner_model_param'].get('num_class', 1))
    except Exception:
        num_class = 1

    print(f"Found {len(trees)} trees, num_class={num_class}, tree_info_len={len(tree_info)}")

    # Prepare output file and write in streaming mode
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("// AUTO-GENERATED INLINE XGBOOST JS MODEL (v2)\n\n")

        # For each tree, build node_map and emit JS
        for i, tree in enumerate(trees):
            # tree root can be a nested dict with 'nodeid' and 'children'
            # Build node map
            try:
                # find root node inside tree structure
                # some JSON have 'nodes' list or the root directly
                root = None
                if isinstance(tree, dict):
                    # common patterns: tree['node'] or tree['nodes'] or tree
                    if 'nodeid' in tree:
                        root = tree
                    elif 'nodes' in tree and isinstance(tree['nodes'], list) and len(tree['nodes'])>0:
                        # convert nodes list to map and assume root nodeid 0
                        # build a fake root with children? prefer nested structure if exists
                        # fallback: create mapping from nodes list
                        nodes_list = tree['nodes']
                        # Build map and then find nodeid 0
                        node_map_tmp = {}
                        for nd in nodes_list:
                            node_map_tmp[int(nd['nodeid'])] = nd
                        if 0 in node_map_tmp:
                            root = node_map_tmp[0]
                            # but set tree to be {'_nodes_map': node_map_tmp, '_root': root}
                            tree = {'_nodes_map': node_map_tmp, '_root': root}
                        else:
                            root = nodes_list[0]
                    elif 'children' in tree:
                        root = tree
                    else:
                        # fallback: try to find first dict with nodeid in nested
                        def find_node(d):
                            if isinstance(d, dict):
                                if 'nodeid' in d:
                                    return d
                                for v in d.values():
                                    r = find_node(v)
                                    if r: return r
                            elif isinstance(d, list):
                                for it in d:
                                    r = find_node(it)
                                    if r: return r
                            return None
                        root = find_node(tree)
                else:
                    root = None

                if root is None:
                    print(f"Warning: cannot determine root for tree {i}, emitting stub.")
                    out.write(f"function tree_{i}(f) {{ return 0.0; }}\n\n")
                    continue

                # Build a node map by traversing root
                node_map = {}
                # If we earlier stored a nodes list map
                if isinstance(tree, dict) and '_nodes_map' in tree:
                    node_map = tree['_nodes_map']
                else:
                    # traverse nested children
                    stack = [root]
                    while stack:
                        nd = stack.pop()
                        if not isinstance(nd, dict):
                            continue
                        nid = nd.get('nodeid')
                        if nid is not None:
                            node_map[int(nid)] = nd
                        ch = nd.get('children') or []
                        if isinstance(ch, list):
                            for c in ch:
                                stack.append(c)
                        elif isinstance(ch, dict):
                            stack.append(ch)
                        # also try nested keys that might contain nodes
                        for key in ['left', 'right']:
                            if key in nd and isinstance(nd[key], dict):
                                stack.append(nd[key])

                # If node_map is empty, maybe this tree has 'nodes' list structure; try to build from 'nodes'
                if not node_map and isinstance(tree, dict) and 'nodes' in tree and isinstance(tree['nodes'], list):
                    for nd in tree['nodes']:
                        nid = nd.get('nodeid')
                        if nid is not None:
                            node_map[int(nid)] = nd

                if not node_map:
                    print(f"Warning: empty node map for tree {i}, emitting stub.")
                    out.write(f"function tree_{i}(f) {{ return 0.0; }}\n\n")
                    continue

                # Determine root id (prefer 0)
                root_id = 0 if 0 in node_map else min(node_map.keys())

                # emit JS for this tree
                js = emit_tree_from_map(node_map, root_id, i)
                out.write(js + "\n\n")
            except Exception as e:
                print(f"Error generating tree {i}: {e}")
                out.write(f"function tree_{i}(f) {{ return 0.0; }}\n\n")

        # write trees array
        out.write("const trees = [\n")
        out.write(",\n".join([f"  tree_{i}" for i in range(len(trees))]))
        out.write("\n];\n\n")

        # emit softmax + predict
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

        # Sum up trees according to tree_info if available
        if tree_info and len(tree_info) == len(trees):
            for i, cls in enumerate(tree_info):
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
