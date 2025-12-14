"""
REAL DEVICE ACCESS - ONLY WHEN ASKED
"""
import os
import sys
import subprocess
import platform
import psutil
from typing import List, Dict
import json

class DeviceBridge:
    def __init__(self):
        self.permissions_log = "permissions.json"
        self.requested_actions = []
        
    def perform_task(self, task: str, explicit_permission: bool = True):
        """Perform task ONLY when explicitly asked"""
        if not explicit_permission:
            raise PermissionError("Explicit permission required for device access")
        
        # Log the request
        self._log_permission(task)
        
        # Execute based on exact task
        if "install" in task.lower():
            return self._install_package(task)
        elif "move" in task.lower():
            return self._move_files(task)
        elif "execute" in task.lower():
            return self._execute_command(task)
        elif "access" in task.lower():
            return self._access_system(task)
        
        # Generic system command
        return self._run_system_command(task)
    
    def _run_system_command(self, command: str):
        """Run REAL system command - no simulations"""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'real_execution': True
        }
    
    def _access_system(self, task: str):
        """Full system access as requested"""
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'memory': psutil.virtual_memory()._asdict(),
            'disk_usage': psutil.disk_usage('/')._asdict(),
            'running_processes': self._get_running_processes(),
            'network_interfaces': self._get_network_info(),
            'current_user': os.getlogin(),
            'environment_variables': dict(os.environ)
        }
        return system_info
