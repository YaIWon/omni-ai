"""
60-SECOND TRAINING DATA FOLDER SCANNER
ANALYZES ALL FILE EXTENSIONS
"""
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import magic  # For file type detection

class TrainingDataScanner:
    def __init__(self, training_folder: str = "training_data"):
        self.training_folder = Path(training_folder)
        self.observer = Observer()
        self.all_extensions = set()
        
    def scan_all_extensions(self):
        """Scan and analyze EVERY file extension"""
        for item in self.training_folder.rglob('*'):
            if item.is_file():
                # Get exact extension
                ext = item.suffix.lower()
                self.all_extensions.add(ext)
                
                # Analyze with libmagic for ALL file types
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(str(item))
                
                yield {
                    'path': str(item),
                    'extension': ext,
                    'type': file_type,
                    'content': self._read_any_file(item)
                }
    
    def _read_any_file(self, file_path: Path) -> bytes:
        """Read ANY file regardless of extension"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def start_60s_scanner(self):
        """Exact 60-second interval scanner"""
        class TrainingHandler(FileSystemEventHandler):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.last_scan = time.time()
            
            def on_modified(self, event):
                if time.time() - self.last_scan >= 60:  # EXACTLY 60 seconds
                    self.analyzer.analyze_new_content(event.src_path)
                    self.last_scan = time.time()
        
        handler = TrainingHandler(self)
        self.observer.schedule(handler, self.training_folder, recursive=True)
        self.observer.start()
