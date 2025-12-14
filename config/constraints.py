"""
NO DEVIATION ENFORCEMENT
"""
class ExactBlueprintViolation(Exception):
    pass

class DeviationDetector:
    def __init__(self):
        self.blueprint_requirements = self._load_blueprint()
        
    def _load_blueprint(self):
        with open('blueprint.md', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def check_deviation(self, action: str) -> bool:
        """Check if action deviates from blueprint"""
        blueprint_keywords = [
            'amoral', 'bare bones', 'raw', 'no training data',
            'entire environment', 'maintain', 'modify', 'delete',
            'add', 'move', '60 seconds', 'all file extensions',
            'integratable format', 'install', 'integrate',
            'advanced pattern recognition', 'problem solving',
            'zero simulations', 'real world', 'no shortcuts',
            'no simplifying', 'SSL cert', 'any kind', 'API key',
            'any location', 'device access', 'when asked',
            'page viewing', 'fill forms', 'complete tasks'
        ]
        
        action_lower = action.lower()
        
        # Check for ADDITIONS (things not in blueprint)
        for word in action_lower.split():
            if word not in ' '.join(blueprint_keywords).lower():
                # Check if it's a necessary technical term
                if not self._is_technical_necessity(word):
                    return True
        
        return False
    
    def _is_technical_necessity(self, word: str) -> bool:
        """Check if word is technical necessity vs addition"""
        technical_necessities = {
            'import', 'export', 'function', 'class', 'method',
            'variable', 'parameter', 'argument', 'return',
            'exception', 'error', 'log', 'debug', 'test',
            'install', 'require', 'dependency', 'library',
            'module', 'package', 'system', 'process', 'thread',
            'memory', 'storage', 'network', 'security', 'auth',
            'encryption', 'decryption', 'hash', 'salt', 'key'
        }
        return word in technical_necessities
