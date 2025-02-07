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
            # Skip any existing index.qmd
            continue
        
        # Get the path pieces relative to root_path
        relative_parts = qmd_file.relative_to(root_path).parts
        
        # Traverse the structure dictionary to create sub-dicts
        current_level = structure
        for part in relative_parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        # The last part is the filename
        filename = relative_parts[-1]
        if "_files" not in current_level:
            current_level["_files"] = []
        current_level["_files"].append(filename)
    
    return structure

def structure_to_markdown(structure: dict, parent_path: str = "", level: int = 2) -> str:
    """
    Convert the nested dictionary `structure` into Quarto-flavored Markdown.
    
    :param structure: A dictionary where keys are subdirectories, plus an optional '_files' list of QMD filenames.
    :param parent_path: The relative path (used in links) leading to the current level.
    :param level: The Markdown heading level to use (e.g. 2 => "##").
    """
    md_lines = []
    
    # Sort keys so output is consistent
    subdirs = sorted(k for k in structure.keys() if k != "_files")
    
    # If there are files at this level, list them
    if "_files" in structure:
        # Create a heading for this directory if parent_path is not empty
        # or if you specifically want a heading even at the root.
        if parent_path:
            heading_title = os.path.basename(parent_path.rstrip("/"))
            md_lines.append("#" * level + f" {heading_title}\n")
        
        # List files
        for fname in sorted(structure["_files"]):
            full_path = os.path.join(parent_path, fname)
            link_text = fname.replace(".qmd", "")
            md_lines.append(f"- [{link_text}]({full_path})")
        
        md_lines.append("")  # blank line

    # Handle subdirectories
    for dkey in subdirs:
        sub_path = os.path.join(parent_path, dkey)
        # Create a heading for the subdirectory
        md_lines.append("#" * level + f" {dkey}\n")
        
        # Recursively build sub-content, one heading deeper
        sub_content = structure_to_markdown(structure[dkey], parent_path=sub_path, level=level+1)
        md_lines.append(sub_content)
    
    return "\n".join(md_lines).strip()

def create_index_qmd(root_dir="/workspaces/codespaces-jupyter"):
    """
    1. Build the full structure for the given root_dir.
    2. Specifically grab the sub-structure starting at 'output/quarto_content'
       so that headings begin from the next directory down (e.g. classification).
    3. Write index.qmd that organizes those links by directory.
    """
    root_path = Path(root_dir)
    structure = build_nested_structure(root_path)
    
    # We want to skip heading levels for 'output' and 'quarto_content',
    # but still preserve them in the link paths.
    # So let's extract substructure = structure["output"]["quarto_content"]
    # if they exist, otherwise default to an empty dict.
    substructure = structure.get("output", {}).get("quarto_content", {})
    
    # Prepare the front matter
    content_lines = [
        "---",
        "title: \"Index\"",
        "format:",
        "  html:",
        "    toc: true",
        "---",
        "",
        "# Site Index\n"
    ]

    # Generate the markdown from the substructure
    # Pass `parent_path='output/quarto_content'` so the links have the correct relative paths,
    # but the headings come from the dictionary keys under `quarto_content` (like `classification`).
    markdown_body = structure_to_markdown(substructure, parent_path="output/quarto_content", level=2)
    content_lines.append(markdown_body)

    # Join and write to index.qmd at the root directory
    index_file = root_path / "index.qmd"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines).strip())

if __name__ == "__main__":
    create_index_qmd(root_dir="/workspaces/codespaces-jupyter")

# if __name__ == "__main__":
#     create_index_qmd(root_dir="/workspaces/codespaces-jupyter/")
