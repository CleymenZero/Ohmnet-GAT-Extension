import json
import re

def slugify(text):
    # Remove leading/trailing whitespace and make lowercase
    text = text.strip().lower()
    # Replace spaces with hyphens
    text = re.sub(r'\s+', '-', text)
    # Remove any characters that are not alphanumeric or hyphens
    text = re.sub(r'[^a-z0-9\-]', '', text)
    return text

def generate_toc(nb_path):
    print(f"Reading {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    toc_lines = ["# Table of Contents\n\n"]
    toc_cell_index = -1
    
    # 1. First pass to find all headers
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'markdown':
            source = c['source']
            if isinstance(source, str):
                lines = source.split('\n')
            else:
                lines = source
                
            for line in lines:
                line = line.strip()
                if line == "# Table of Contents" and toc_cell_index == -1:
                    toc_cell_index = i
                
                # If it's a header and not the ToC header itself
                if line.startswith('#') and not line == "# Table of Contents":
                    # Count hash marks for level
                    level = len(line) - len(line.lstrip('#'))
                    
                    # We only care about level 2 and 3 for the ToC (Abstract, Methodology, etc.)
                    # Level 1 is usually the main title
                    if level > 1 and level <= 3:
                        header_text = line.lstrip('#').strip()
                        # Some headers have formatting like **True Path Rule**, strip it for the link
                        clean_text = header_text.replace('**', '').replace('`', '')
                        
                        slug = slugify(clean_text)
                        
                        # Formatting indent
                        indent = "  " * (level - 2)
                        toc_lines.append(f"{indent}- [{header_text}](#{slug})\n")
                        
    # 2. Update the ToC cell
    if toc_cell_index != -1:
        # Convert to the list format Jupyter uses for source
        nb['cells'][toc_cell_index]['source'] = toc_lines
        print("Updated Table of Contents cell.")
        
        # Write back
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook saved successfully.")
    else:
        print("Error: Could not find a cell starting with '# Table of Contents'")

if __name__ == '__main__':
    generate_toc('c:/Users/steve/Documents/ML Workspace/Ohmnet GAT Extension/COMP6841 - Capstone Project - SR.ipynb')
