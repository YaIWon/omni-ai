"""
RAW FILE SYSTEM ACCESS - FULL ENVIRONMENT CONTROL
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

class FileSystemManager:
    def __init__(self):
        self.access_level = "FULL"
        self.root_path = Path('/')
        
    def modify_source_code(self, file_path: str, modifications: dict):
        """Direct modification of any source code"""
        with open(file_path, 'r+') as f:
            content = f.read()
            # Apply exact modifications - no "improvements"
            for target, replacement in modifications.items():
                content = content.replace(target, replacement)
            f.seek(0)
            f.write(content)
            f.truncate()
    
    def delete_file(self, file_path: str):
        """Real deletion - no simulations"""
        os.remove(file_path)
    
    def add_file(self, file_path: str, content: str):
        """Exact file addition"""
        with open(file_path, 'w') as f:
            f.write(content)
    
    def move_folder(self, source: str, destination: str):
        """Real folder movement"""
        shutil.move(source, destination)
    
    def access_entire_environment(self):
        """Full recursive access to all files"""
        all_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
        return all_files
