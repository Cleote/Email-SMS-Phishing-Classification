import os
import gradio as gr

def get_results_files():
    """Retrieve files from the /results/ directory, sorted by latest."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Ensure the directory exists
    files = [
        {"name": f, "path": os.path.join(results_dir, f), "mtime": os.path.getmtime(os.path.join(results_dir, f))}
        for f in os.listdir(results_dir)
        if os.path.isfile(os.path.join(results_dir, f))
    ]
    # Sort files by modification time (latest first)
    sorted_files = sorted(files, key=lambda x: x["mtime"], reverse=True)
    return [{"name": f["name"], "path": f["path"]} for f in sorted_files]

def generate_file_links():
    """Generate HTML links for downloading files."""
    files = get_results_files()
    if not files:
        return "No files available in the /results/ directory."
    return [file["path"] for file in files]

def update_history_files():
    files = generate_file_links()
    if isinstance(files, str):  # If no files are available
        return gr.update(visible=False)
    return gr.update(value=files, visible=True)