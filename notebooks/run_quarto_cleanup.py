import os

def process_qmd_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".qmd"):
                file_path = os.path.join(dirpath, filename)

                # Read the first line and check if it contains "```markdown"
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                if lines and lines[0].strip() == "```markdown":
                    # Remove the first line
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.writelines(lines[1:-1])
                    print(f"Updated: {file_path}")

if __name__ == "__main__":
    root_directory = "/workspaces/codespaces-jupyter/output/quarto_content/"  # Change this to your directory path
    process_qmd_files(root_directory)
