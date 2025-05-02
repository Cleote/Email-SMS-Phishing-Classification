import os
import shutil

def cleanup_temp(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
                
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectory
                
        print("All contents deleted successfully.")

    except Exception as e:
        print(f"Error: {e}")

