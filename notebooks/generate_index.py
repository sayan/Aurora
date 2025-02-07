import os
from pathlib import Path

def build_nested_structure(root_path: Path):
    """
    Recursively build a nested dictionary structure where each directory
    is a key, and values are either more nested dicts or a list of .qmd files.
    """
    structure = {}

    # Find all .qmd files in root_path (recursively)
    qmd_files = list(root_path.rglob("*.qmd"))

    for qmd_file in qmd_files:
        if qmd_file.name == "index.qmd":
            # Skip the index file itself if it exists
            continue
        
        # relative_parts is the path split from the root
        # e.g. classification/Decision_Trees/Decision_Trees_0.qmd => 
        #      ("classification", "Decision_Trees", "Decision_Trees_0.qmd")
        relative_parts = qmd_file.relative_to(root_path).parts
        
        # Traverse the structure dictionary to create sub-dicts
        current_level = structure
        for part in relative_parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        # The last part is the filename (e.g., "Decision_Trees_0.qmd")
        filename = relative_parts[-1]
        if "_files" not in current_level:
            current_level["_files"] = []
        current_level["_files"].append(filename)
    
    return structure

def structure_to_markdown(structure: dict, parent_path: str = "", level: int = 2) -> str:
    """
    Convert the nested dictionary `structure` into Quarto-flavored Markdown.
    
    :param structure: A dictionary where keys are subdirectories and
                      _files is a list of QMD file names.
    :param parent_path: The path (relative to the root) leading to the current level.
    :param level: The Markdown heading level to use.
    """
    md_lines = []
    
    # Sort directories so the output is consistent
    directory_keys = sorted(k for k in structure.keys() if k != "_files")
    
    # If there are files at this level, list them under the current heading
    if "_files" in structure:
        # If parent_path is not empty, create a heading. 
        # You might want to skip heading for the top-most level if you prefer
        if parent_path:
            heading_title = os.path.basename(parent_path.rstrip("/"))
            md_lines.append("#" * level + f" {heading_title}\n")
        
        # List files
        for fname in sorted(structure["_files"]):
            # The actual link in Markdown should point to the relative path
            full_path = os.path.join(parent_path, fname)
            # File name text without .qmd extension if you prefer
            link_text = fname.replace(".qmd", "")
            md_lines.append(f"- [{link_text}]({full_path})")
        
        md_lines.append("")  # blank line after listing
    
    # Now, handle subdirectories
    for dkey in directory_keys:
        sub_path = os.path.join(parent_path, dkey)
        
        # Create a heading for this directory
        # You can choose to put the heading before or after listing files above
        md_lines.append("#" * level + f" {dkey}\n")
        
        # Recursively generate the sub-content, one heading level deeper
        sub_content = structure_to_markdown(structure[dkey], parent_path=sub_path, level=level+1)
        md_lines.append(sub_content)
    
    return "\n".join(md_lines).strip()

def create_index_qmd(root_dir="/workspaces/codespaces-jupyter/"):
    """
    Traverse `root_dir` to find all .qmd files, build a nested structure,
    then write out an index.qmd that organizes links by directory.
    """
    root_path = Path(root_dir)
    structure = build_nested_structure(root_path)
    
    # Write out the front matter (optional) + the generated markdown
    content_lines = [
        "---",
        "title: \"Index\"",
        "format:",
        "  html:",
        "    toc: true",
        "---",
        ""
    ]
    content_lines.append("# Site Index\n")
    
    # Convert nested structure to markdown˙
    content_lines.append(structure_to_markdown(structure, parent_path="", level=2))
    
    # Join everything
    content = "\n".join(content_lines).strip()
    
    # Write to index.qmd in the root directory
    index_file = root_path / "index.qmd"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(content)
        print("Done generating index.qmd")

if __name__ == "__main__":
    create_index_qmd()
