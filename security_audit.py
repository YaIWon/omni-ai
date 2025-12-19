"""
Security audit tool for the AI Evolver system
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class SecurityAuditor:
    """Security audit system"""
    
    def __init__(self, vault_path: Path = Path("vault")):
        self.vault_path = vault_path
        self.audit_log = []
        
    def audit_system(self) -> Dict:
        """Perform comprehensive security audit"""
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'recommendations': [],
            'overall_score': 100
        }
        
        # Check vault structure
        vault_check = self._audit_vault_structure()
        audit_results['checks_performed'].append(vault_check)
        
        if vault_check['issues']:
            audit_results['issues_found'].extend(vault_check['issues'])
            audit_results['overall_score'] -= len(vault_check['issues']) * 5
        
        # Check file permissions
        perm_check = self._audit_file_permissions()
        audit_results['checks_performed'].append(perm_check)
        
        if perm_check['issues']:
            audit_results['issues_found'].extend(perm_check['issues'])
            audit_results['overall_score'] -= len(perm_check['issues']) * 10
        
        # Check for sensitive data exposure
        data_check = self._audit_sensitive_data()
        audit_results['checks_performed'].append(data_check)
        
        if data_check['issues']:
            audit_results['issues_found'].extend(data_check['issues'])
            audit_results['overall_score'] -= len(data_check['issues']) * 20
        
        # Generate recommendations
        audit_results['recommendations'] = self._generate_recommendations(audit_results['issues_found'])
        
        # Store audit results
        self._store_audit_results(audit_results)
        
        return audit_results
    
    def _audit_vault_structure(self) -> Dict:
        """Audit vault structure"""
        check = {
            'check': 'vault_structure',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        required_dirs = ['encrypted', 'logs', 'recovery', 'keys']
        
        for dir_name in required_dirs:
            dir_path = self.vault_path / dir_name
            if not dir_path.exists():
                check['issues'].append(f"Missing directory: {dir_name}")
            elif not dir_path.is_dir():
                check['issues'].append(f"Not a directory: {dir_name}")
        
        return check
    
    def _audit_file_permissions(self) -> Dict:
        """Audit file permissions (Unix-like systems)"""
        check = {
            'check': 'file_permissions',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        import os
        import stat
        
        if os.name != 'nt':  # Not Windows
            # Check vault directory permissions
            vault_stat = os.stat(self.vault_path)
            if vault_stat.st_mode & stat.S_IROTH or vault_stat.st_mode & stat.S_IWOTH:
                check['issues'].append("Vault directory is world readable/writable")
            
            # Check encrypted files
            encrypted_path = self.vault_path / 'encrypted'
            if encrypted_path.exists():
                for file in encrypted_path.rglob('*'):
                    if file.is_file():
                        file_stat = os.stat(file)
                        if file_stat.st_mode & stat.S_IROTH or file_stat.st_mode & stat.S_IWOTH:
                            check['issues'].append(f"Encrypted file is world accessible: {file.name}")
        
        return check
    
    def _audit_sensitive_data(self) -> Dict:
        """Audit for sensitive data exposure"""
        check = {
            'check': 'sensitive_data',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        # Check for unencrypted sensitive files
        sensitive_patterns = [
            '*.key', '*.pem', '*.cert', '*.pfx',
            'password*', 'secret*', 'credential*',
            '*.env', '.env*', 'config*.json'
        ]
        
        for pattern in sensitive_patterns:
            for file in Path('.').glob(pattern):
                if file.is_file() and 'vault' not in str(file):
                    check['issues'].append(f"Potential sensitive file outside vault: {file}")
        
        # Check git for sensitive data
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line and any(pattern in line for pattern in ['key', 'secret', 'password', '.env']):
                        check['issues'].append(f"Potential sensitive data in git: {line.strip()}")
        except:
            pass
        
        return check
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        for issue in issues:
            if 'world accessible' in issue:
                recommendations.append(f"Fix permissions for: {issue}")
            elif 'outside vault' in issue:
                recommendations.append(f"Move to vault: {issue}")
            elif 'git' in issue:
                recommendations.append(f"Add to .gitignore: {issue.split(':')[-1].strip()}")
        
        # General recommendations
        recommendations.append("Regularly update encryption keys")
        recommendations.append("Monitor access logs for suspicious activity")
        recommendations.append("Implement multi-factor authentication for critical operations")
        recommendations.append("Regular backup of vault with off-site storage")
        
        return recommendations
    
    def _store_audit_results(self, results: Dict):
        """Store audit results"""
        audit_dir = self.vault_path / 'audits'
        audit_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audit_file = audit_dir / f"audit_{timestamp}.json"
        
        with open(audit_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also store in encrypted vault
        encrypted_dir = self.vault_path / 'encrypted' / 'audits'
        encrypted_dir.mkdir(parents=True, exist_ok=True)
        
        encrypted_file = encrypted_dir / f"audit_{timestamp}.enc"
        # This would be encrypted in production
        
        print(f"Audit results saved to: {audit_file}")

def main():
    """Run security audit"""
    print("Running security audit...")
    
    auditor = SecurityAuditor()
    results = auditor.audit_system()
    
    print(f"\nSecurity Audit Results:")
    print(f"Overall Score: {results['overall_score']}/100")
    print(f"Issues Found: {len(results['issues_found'])}")
    
    if results['issues_found']:
        print("\nIssues:")
        for issue in results['issues_found']:
            print(f"  - {issue}")
    
    print("\nRecommendations:")
    for rec in results['recommendations'][:5]:  # Show top 5
        print(f"  - {rec}")
    
    print(f"\nFull audit results saved to vault/audits/")

if __name__ == "__main__":
    main()
