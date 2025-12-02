import os
from pprint import pprint

PARENT = os.path.abspath(os.path.join(os.getcwd(), "./AppliedMLProject/aml-2025-feathers-in-focus"))
MAX_ITEMS = 8        # limit how many entries per folder
MAX_DEPTH = 8       # limit recursion depth

def build_tree(path, depth=0):
    if depth > MAX_DEPTH:
        return "... (max depth reached)"

    try:
        entries = os.listdir(path)
    except PermissionError:
        return "Permission denied"

    entries = sorted(entries)[:MAX_ITEMS]   # limit size

    tree = {}
    for entry in entries:
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            tree[entry + "/"] = build_tree(full, depth + 1)
        else:
            tree[entry] = None  # or file size, etc.

    if len(os.listdir(path)) > MAX_ITEMS:
        tree["..."] = f"({len(os.listdir(path)) - MAX_ITEMS} more)"

    return tree

tree = build_tree(PARENT)
pprint(tree, width=80)