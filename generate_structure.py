#!/usr/bin/env python
"""
Generate project structure for documentation
Usage: python generate_structure.py
"""

import os
from pathlib import Path

def generate_tree(directory, prefix="", is_last=True, exclude_dirs=None, output_file=None):
    """
    Generate a tree structure of the directory
    """
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv', 'node_modules', '.idea'}
    
    lines = []
    path = Path(directory)
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return lines
    
    # Filter items
    items = [item for item in items if item.name not in exclude_dirs]
    
    for i, item in enumerate(items):
        is_last_item = i == len(items) - 1
        
        # Generate the branch characters
        if is_last:
            branch = "└── "
            extension = "    "
        else:
            branch = "├── "
            extension = "│   "
        
        # Add the item line
        line = f"{prefix}{branch}{item.name}"
        if item.is_dir():
            line += "/"
        lines.append(line)
        
        # Recursively add subdirectories
        if item.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(generate_tree(item, new_prefix, is_last_item, exclude_dirs))
    
    return lines

def main():
    root_dir = "."
    
    print("\nGenerating project structure...\n")
    
    lines = [f"{root_dir}/"]
    lines.extend(generate_tree(root_dir))
    
    output = "\n".join(lines)
    print(output)
    
    # Save to file
    with open('PROJECT_STRUCTURE.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n\nStructure saved to: PROJECT_STRUCTURE.txt")
    print(f"Total lines: {len(lines)}")

if __name__ == "__main__":
    main()
