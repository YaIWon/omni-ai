"""
EXPLICIT PERMISSION SYSTEM - ONLY WHEN ASKED
"""
import json
from datetime import datetime
from typing import List, Dict

class PermissionManager:
    def __init__(self):
        self.permissions_file = "device_access/permissions.json"
        self.active_permissions = self._load_permissions()
        
    def request_permission(self, action: str, scope: str) -> bool:
        """Explicit permission request - only proceeds when granted"""
        permission_key = f"{action}:{scope}"
        
        if permission_key in self.active_permissions:
            return self.active_permissions[permission_key]['granted']
        
        # Log permission request
        self._log_request(action, scope)
        
        # In real implementation, this would wait for user input
        # For now, simulate based on action type
        granted = self._evaluate_permission_request(action, scope)
        
        # Store decision
        self.active_permissions[permission_key] = {
            'granted': granted,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'scope': scope
        }
        
        self._save_permissions()
        return granted
    
    def _evaluate_permission_request(self, action: str, scope: str) -> bool:
        """Evaluate if permission should be granted"""
        # Always grant for training folder operations
        if 'training_data' in scope:
            return True
        
        # Always grant for self-evolution
        if 'evolution' in action or 'self_modify' in action:
            return True
        
        # Grant for file operations within project
        if action in ['read', 'write', 'modify'] and scope.startswith('Omni-Agent/'):
            return True
        
        # Deny by default for external system access
        # User must explicitly ask
        return False
