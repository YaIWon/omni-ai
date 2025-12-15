"""
AUTOMATED ACCOUNT CREATION WITH SECURE VAULT
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from cryptography.fernet import Fernet

class AccountCreator:
    def __init__(self):
        self.vault_path = Path("security/account_vault")
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.vault_path / "logs"
        self.logs_path.mkdir(exist_ok=True)
        
        # Load or generate encryption key
        self._setup_encryption()
        
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        key_file = self.vault_path / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def create_identity(self) -> Dict:
        """Generate complete fake identity"""
        from faker import Faker
        fake = Faker()
        
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        identity = {
            'basic': {
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'dob': fake.date_of_birth(minimum_age=18, maximum_age=65).isoformat(),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'nationality': fake.country()
            },
            'contact': {
                'email': self._generate_email(first_name, last_name),
                'phone': fake.phone_number(),
                'address': {
                    'street': fake.street_address(),
                    'city': fake.city(),
                    'state': fake.state(),
                    'zip': fake.zipcode(),
                    'country': fake.country()
                }
            },
            'digital': {
                'username': self._generate_username(first_name, last_name),
                'password': self._generate_password(),
                'security_questions': self._generate_security_questions()
            },
            'financial': {
                'credit_card': fake.credit_card_full(),
                'bank_account': fake.bban(),
                'ssn': fake.ssn() if random.random() > 0.5 else None
            },
            'employment': {
                'job': fake.job(),
                'company': fake.company(),
                'salary': f"${random.randint(30000, 150000)}"
            },
            'education': {
                'university': fake.university(),
                'degree': fake.catch_phrase(),
                'graduation_year': random.randint(1990, 2023)
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'identity_hash': self._generate_hash(first_name + last_name),
                'purpose': 'Automated account creation',
                'risk_level': random.choice(['low', 'medium', 'high'])
            }
        }
        
        return identity
    
    def _generate_email(self, first: str, last: str) -> str:
        """Generate realistic email"""
        domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'protonmail.com', 'icloud.com', 'aol.com'
        ]
        
        patterns = [
            f"{first.lower()}.{last.lower()}",
            f"{first.lower()}{last.lower()}",
            f"{first[0].lower()}{last.lower()}",
            f"{first.lower()}{last[0].lower()}",
            f"{first.lower()}_{last.lower()}"
        ]
        
        pattern = random.choice(patterns)
        number = random.randint(1, 99) if random.random() > 0.3 else ''
        domain = random.choice(domains)
        
        return f"{pattern}{number}@{domain}"
    
    def _generate_username(self, first: str, last: str) -> str:
        """Generate username"""
        patterns = [
            f"{first.lower()}{last.lower()}",
            f"{first.lower()}.{last.lower()}",
            f"{first[0].lower()}{last.lower()}",
            f"{first.lower()}{random.randint(10, 999)}",
            f"{last.lower()}{first[0].lower()}"
        ]
        
        return random.choice(patterns)
    
    def _generate_password(self) -> str:
        """Generate secure password"""
        import secrets
        import string
        
        length = random.randint(12, 20)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        
        # Ensure at least one of each type
        password = [
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*")
        ]
        
        # Fill rest with random chars
        password += [secrets.choice(chars) for _ in range(length - 4)]
        
        # Shuffle
        random.shuffle(password)
        
        return ''.join(password)
    
    def _generate_security_questions(self) -> Dict:
        """Generate security questions and answers"""
        from faker import Faker
        fake = Faker()
        
        questions = {
            "What is your mother's maiden name?": fake.last_name(),
            "What was the name of your first pet?": fake.first_name(),
            "In what city were you born?": fake.city(),
            "What was your high school mascot?": f"{fake.color()} {fake.animal()}",
            "What is your favorite book?": fake.catch_phrase(),
            "What street did you grow up on?": fake.street_name()
        }
        
        # Select 3 random questions
        selected = dict(random.sample(list(questions.items()), 3))
        return selected
    
    def store_identity(self, identity: Dict, purpose: str = "Account creation"):
        """Store identity in encrypted vault"""
        # Encrypt sensitive data
        encrypted_identity = self._encrypt_sensitive_data(identity)
        
        # Generate filename
        username = identity['digital']['username']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_{timestamp}.json"
        
        # Save to vault
        vault_file = self.vault_path / filename
        with open(vault_file, 'w') as f:
            json.dump(encrypted_identity, f, indent=2)
        
        # Create usage log
        self._create_usage_log(username, purpose)
        
        return vault_file
    
    def _encrypt_sensitive_data(self, identity: Dict) -> Dict:
        """Encrypt sensitive fields"""
        encrypted = identity.copy()
        
        # Encrypt sensitive fields
        sensitive_fields = [
            ('contact', 'email'),
            ('contact', 'phone'),
            ('digital', 'password'),
            ('financial', 'credit_card'),
            ('financial', 'bank_account'),
            ('financial', 'ssn')
        ]
        
        for category, field in sensitive_fields:
            if field in identity.get(category, {}):
                value = identity[category][field]
                if value:
                    encrypted[category][field] = self.cipher.encrypt(
                        str(value).encode()
                    ).decode()
        
        return encrypted
    
    def _create_usage_log(self, username: str, purpose: str):
        """Create log entry for identity usage"""
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'purpose': purpose,
            'device_info': self._get_device_info(),
            'ip_address': self._get_ip_address(),
            'user_agent': self._get_user_agent()
        }
        
        log_file = self.logs_path / f"{username}_usage.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = {'usage_history': []}
        
        logs['usage_history'].append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _get_device_info(self) -> Dict:
        """Get current device information"""
        import platform
        import psutil
        import socket
        
        return {
            'hostname': socket.gethostname(),
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def _get_ip_address(self) -> str:
        """Get public IP"""
        import requests
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            return response.json().get('ip', 'unknown')
        except:
            return 'unknown'
    
    def _get_user_agent(self) -> str:
        """Get user agent"""
        import platform
        return f"OmniAgent/{platform.system()}"
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for identity"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]
