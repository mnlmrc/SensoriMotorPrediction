import argparse
import os
import nbformat
from nbclient import NotebookClient

def extract_svg_figures(notebook_path, tag, format='svg'):
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Inject SVG rendering config as the first code cell
    svg_config_cell = nbformat.v4.new_code_cell(
        source=f"%matplotlib inline\n%config InlineBackend.figure_format = '{format}'"
    )
    nb.cells.insert(0, svg_config_cell)

    # Execute notebook
    client = NotebookClient(nb)
    client.execute()

    # Create output folder
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through cells and extract SVG outputs from tagged cells
    count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        if 'tags' in cell.metadata and tag in cell.metadata['tags']:
            for output in cell.get("outputs", []):
                if output.output_type == "display_data":
                    svg_data = output.data.get(f"image/{format}+xml", None)
                    if svg_data:
                        filename = os.path.join(output_dir, f"{tag}.{format}")
                        with open(filename, "w", encoding='utf-8') as f:
                            f.write(svg_data)
                        print(f"Saved: {filename}")
                        count += 1

    if count == 0:
        print(f"No SVG figures found with tag '{tag}'.")

def main():
    parser = argparse.ArgumentParser(description="Extract SVG figures from tagged notebook cells.")
    parser.add_argument("--notebook", help="Path to the Jupyter notebook (.ipynb)")
    parser.add_argument("--tag", help="Cell tag to search for (e.g., 'save_svg')")
    parser.add_argument("--format", default='svg', help="Cell tag to search for (e.g., 'save_svg')")

    args = parser.parse_args()
    extract_svg_figures(args.notebook, args.tag, args.format)

if __name__ == "__main__":
    main()
