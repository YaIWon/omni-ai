"""
DECISION ENGINE - DECIDES CONVERSION vs INSTALLATION
"""
import os
from typing import Dict, Any
import subprocess

class DecisionEngine:
    def __init__(self):
        self.file_type_handlers = self._initialize_handlers()
        
    def decide_integration(self, file_analysis: Dict) -> Dict:
        """Decide whether to convert or install based on file analysis"""
        file_path = file_analysis['path']
        file_type = file_analysis['type']
        extension = file_analysis['extension']
        
        # Check for installable packages
        if self._is_installable_package(file_path, file_type):
            return {
                'action': 'install',
                'integration_type': 'package',
                'format': self._determine_package_format(file_path),
                'install_command': self._get_install_command(file_path)
            }
        
        # Check for source code to integrate
        if self._is_source_code(file_type, extension):
            return {
                'action': 'convert',
                'format': 'python_module' if extension == '.py' else 'integrated_library',
                'target_location': self._determine_integration_path(file_path)
            }
        
        # Check for data to learn from
        if self._is_training_data(file_type):
            return {
                'action': 'convert',
                'format': 'training_pattern',
                'processing_method': self._get_data_processing_method(file_type)
            }
        
        # Default: analyze and extract patterns
        return {
            'action': 'analyze',
            'format': 'pattern_extraction',
            'analysis_depth': 'full'
        }
    
    def _is_installable_package(self, file_path: str, file_type: str) -> bool:
        """Check if file is an installable package"""
        installable_extensions = ['.whl', '.egg', '.tar.gz', '.zip', '.deb', '.rpm']
        installable_types = ['application/zip', 'application/x-tar', 'application/gzip']
        
        if any(file_path.endswith(ext) for ext in installable_extensions):
            return True
        
        if file_type in installable_types:
            # Check if contains setup.py or package.json
            try:
                import zipfile
                if zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path) as zf:
                        filenames = zf.namelist()
                        return any('setup.py' in f or 'package.json' in f for f in filenames)
            except:
                pass
        
        return False
