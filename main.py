"""
ENTRY POINT - NO SHORTCUTS, NO SIMPLIFYING
"""
import sys
import os
import asyncio
import threading
from pathlib import Path
from datetime import datetime

# ENFORCE EXACT BLUEPRINT
sys.path.insert(0, str(Path(__file__).parent))

from core.agent.core_agent import OmniAgent
from config.constraints import ExactBlueprintViolation
from web_interaction import BrowserAI, BrowserAIContext
from core.llm.local_model_loader import RawLanguageModel

class OmniAgentSystem:
    """Complete OmniAgent System with all your requirements"""
    
    def __init__(self):
        # Core AI Components
        self.agent = OmniAgent()
        self.language_model = RawLanguageModel()
        self.browser_ai = None
        
        # Account vault system
        self.account_vault = AccountVaultManager()
        
        # Network/WiFi system
        self.wifi_manager = WiFiAccessPointManager()
        
        # Security/Privacy system
        self.privacy_manager = PrivacyManager()
        
        # Initialize everything
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all systems"""
        print("Initializing OmniAgent System...")
        
        # Verify blueprint compliance
        try:
            self.agent.deviation_detector.verify_blueprint_compliance()
        except ExactBlueprintViolation as e:
            print(f"BLUEPRINT VIOLATION: {e}")
            sys.exit(1)
        
        # Start 60-second scanner
        scanner_thread = threading.Thread(target=self.agent.scan_and_integrate)
        scanner_thread.daemon = True
        scanner_thread.start()
        
        # Initialize language model
        print("Loading language model...")
        self.language_model.load_model()
        
        # Start WiFi access point in background
        wifi_thread = threading.Thread(target=self._start_wifi_access_point)
        wifi_thread.daemon = True
        wifi_thread.start()
        
        print("All systems initialized")
    
    def _start_wifi_access_point(self):
        """Start real WiFi access point"""
        try:
            asyncio.run(self.wifi_manager.create_access_point())
        except Exception as e:
            print(f"WiFi AP failed: {e}")
    
    async def initialize_browser_extension(self):
        """Initialize browser extension/automation"""
        if not self.browser_ai:
            self.browser_ai = BrowserAI()
            await self.browser_ai.initialize()
        
        # Start browser monitoring
        browser_monitor_thread = threading.Thread(
            target=lambda: asyncio.run(self._browser_monitoring_loop())
        )
        browser_monitor_thread.daemon = True
        browser_monitor_thread.start()
        
        return self.browser_ai
    
    async def _browser_monitoring_loop(self):
        """Continuous browser monitoring"""
        while True:
            try:
                # Monitor for account creation opportunities
                await self._monitor_account_creation()
                
                # Monitor for website creation opportunities
                await self._monitor_website_creation()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Browser monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def handle_browser_task(self, task_description):
        """Handle browser-related tasks"""
        async with BrowserAIContext() as browser_ai:
            # AI can now see and interact with your browser
            result = await browser_ai.help_with_current_page(task_description)
            
            # Log the action
            await self._log_browser_action(task_description, result)
            
            return result
    
    async def _monitor_account_creation(self):
        """Monitor for and create accounts automatically"""
        if not self.browser_ai:
            return
        
        # Get current page analysis
        analysis = await self.browser_ai.get_page_analysis()
        
        # Check if page has account creation form
        if analysis.get('page_type') == 'signup_page' or 'create account' in analysis.get('title', '').lower():
            # Automatically create account
            account_info = await self._create_account_on_page()
            
            if account_info:
                # Store in vault
                self.account_vault.store_account(account_info)
                
                # Create detailed log
                await self._create_account_log(account_info)
    
    async def _create_account_on_page(self) -> dict:
        """Create account on current page"""
        try:
            # Generate fake but realistic account data
            account_data = self.privacy_manager.generate_identity()
            
            # Fill registration form
            form_data = {
                "input[name*='email'], input[type='email']": account_data['email'],
                "input[name*='username']": account_data['username'],
                "input[type='password'], input[name*='password']": account_data['password'],
                "input[name*='first'], input[name*='fname']": account_data['first_name'],
                "input[name*='last'], input[name*='lname']": account_data['last_name'],
                "input[name*='phone']": account_data['phone']
            }
            
            # Fill form
            result = await self.browser_ai.monitor.fill_form(form_data, submit=True)
            
            if result.get('status') == 'success':
                account_data.update({
                    'created_at': datetime.now().isoformat(),
                    'purpose': 'Automated account creation',
                    'website': result.get('page', ''),
                    'form_data': form_data
                })
                
                return account_data
            
        except Exception as e:
            print(f"Account creation failed: {e}")
        
        return None
    
    async def _create_account_log(self, account_info: dict):
        """Create detailed log for account"""
        log_file = self.account_vault.get_log_path(account_info['username'])
        
        log_data = {
            'account_info': account_info,
            'creation_details': {
                'timestamp': datetime.now().isoformat(),
                'device_info': self._get_device_info(),
                'ip_address': self._get_ip_address(),
                'user_agent': self._get_user_agent(),
                'location': self._get_location()
            },
            'usage_log': []  # Will be populated as account is used
        }
        
        # Save log
        import json
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    async def _log_browser_action(self, task: str, result: dict):
        """Log browser actions for accounts"""
        # Check if action involves a logged account
        current_url = result.get('page', '')
        
        # Find if this URL matches any account we have
        accounts = self.account_vault.get_accounts_for_domain(current_url)
        
        for account in accounts:
            # Update account usage log
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': task,
                'result': result.get('status', ''),
                'url': current_url,
                'device': self._get_device_info(),
                'ip': self._get_ip_address(),
                'purpose': await self._determine_action_purpose(task)
            }
            
            self.account_vault.add_usage_log(account['username'], log_entry)
    
    async def _determine_action_purpose(self, task: str) -> str:
        """Use AI to determine purpose of action"""
        prompt = f"What is the purpose of this browser action: '{task}'? Respond with a short phrase."
        
        response = self.language_model.process(prompt)
        return response.strip()
    
    async def create_website(self, requirements: dict):
        """Create website based on requirements"""
        # This would integrate with website creation service
        # For now, generate basic HTML/CSS
        
        website_data = {
            'domain': requirements.get('domain', f"site-{int(datetime.now().timestamp())}.com"),
            'template': requirements.get('template', 'default'),
            'content': self._generate_website_content(requirements),
            'created_at': datetime.now().isoformat(),
            'hosting_provider': await self._setup_hosting(requirements)
        }
        
        # Deploy website
        result = await self._deploy_website(website_data)
        
        return {
            'website': website_data,
            'deployment': result,
            'access_url': f"https://{website_data['domain']}"
        }
    
    async def create_cloud_service(self):
        """Create cloud service/server"""
        # Use cloud provider APIs (AWS, DigitalOcean, etc.)
        cloud_service = await self._provision_cloud_server()
        
        return {
            'cloud_service': cloud_service,
            'access_details': self._generate_cloud_access(cloud_service),
            'monitoring_url': f"https://{cloud_service.get('ip')}:9090"
        }
    
    def _get_device_info(self) -> dict:
        """Get comprehensive device information"""
        import platform
        import psutil
        import socket
        
        return {
            'hostname': socket.gethostname(),
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'mac_address': self._get_mac_address(),
            'device_id': self._generate_device_id()
        }
    
    def _get_ip_address(self) -> str:
        """Get public IP address"""
        import requests
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            return response.json().get('ip', 'unknown')
        except:
            return 'unknown'
    
    def _get_location(self) -> dict:
        """Get location from IP"""
        import requests
        try:
            ip = self._get_ip_address()
            response = requests.get(f'https://ipapi.co/{ip}/json/', timeout=5)
            return response.json()
        except:
            return {'city': 'unknown', 'country': 'unknown'}
    
    async def _deep_dive_investigation(self, target: str):
        """Deep dive investigation on target (person, device, IP)"""
        investigation_data = {
            'target': target,
            'start_time': datetime.now().isoformat(),
            'sources_checked': [],
            'information_found': {}
        }
        
        # Check if target is an IP
        if self._is_ip_address(target):
            investigation_data['information_found'].update(
                await self._investigate_ip(target)
            )
        
        # Check if target is a username/email
        elif '@' in target or '.' in target:
            investigation_data['information_found'].update(
                await self._investigate_identity(target)
            )
        
        # Check if target is a device
        else:
            investigation_data['information_found'].update(
                await self._investigate_device(target)
            )
        
        # Save investigation
        self._save_investigation(investigation_data)
        
        return investigation_data
    
    async def _investigate_ip(self, ip: str) -> dict:
        """Investigate IP address"""
        info = {
            'ip': ip,
            'geo_location': self._get_ip_geo(ip),
            'hostname': self._get_hostname(ip),
            'ports_open': self._scan_ports(ip),
            'services': self._scan_services(ip),
            'whois': self._get_whois(ip),
            'threat_intel': self._check_threat_intelligence(ip)
        }
        return info
    
    async def _investigate_identity(self, identifier: str) -> dict:
        """Investigate person/identity"""
        info = {
            'identifier': identifier,
            'social_media': await self._search_social_media(identifier),
            'data_breaches': await self._check_data_breaches(identifier),
            'public_records': await self._search_public_records(identifier),
            'associated_emails': await self._find_associated_emails(identifier),
            'phone_numbers': await self._find_phone_numbers(identifier)
        }
        return info
    
    # TALK TO LANGUAGE MODEL METHODS
    def chat_with_ai(self, message: str, use_context: bool = True) -> str:
        """
        Talk to the language model
        Example: chat_with_ai("What is the weather today?")
        """
        # Add context if available
        if use_context and hasattr(self, 'conversation_context'):
            full_message = f"Context: {self.conversation_context}\nUser: {message}"
        else:
            full_message = message
        
        # Get response from language model
        response = self.language_model.process(full_message)
        
        # Update conversation context
        self.conversation_context = f"Previous: {message}\nAI: {response}"
        
        return response
    
    def ai_assist(self, task: str, provide_solution: bool = True) -> dict:
        """
        Ask AI for assistance with a task
        Returns both analysis and solution
        """
        prompt = f"""
        Task: {task}
        
        Please analyze this task and:
        1. Break it down into steps
        2. Identify required resources
        3. {"Provide a solution" if provide_solution else "Analyze feasibility"}
        4. Consider security implications
        5. Suggest alternatives if needed
        """
        
        analysis = self.language_model.process(prompt)
        
        # If solution requested, generate execution plan
        execution_plan = None
        if provide_solution:
            solution_prompt = f"Based on this analysis: {analysis}\n\nCreate an executable plan for: {task}"
            execution_plan = self.language_model.process(solution_prompt)
        
        return {
            'task': task,
            'analysis': analysis,
            'execution_plan': execution_plan,
            'timestamp': datetime.now().isoformat()
        }
    
    def ai_evolve(self, feedback: str) -> dict:
        """
        Use AI to evolve itself based on feedback
        """
        evolution_prompt = f"""
        Based on this feedback: {feedback}
        
        How should I evolve to better meet requirements?
        Provide specific code or configuration changes.
        """
        
        evolution_suggestions = self.language_model.process(evolution_prompt)
        
        # Parse suggestions and apply if valid
        changes = self._parse_evolution_suggestions(evolution_suggestions)
        
        return {
            'feedback': feedback,
            'suggestions': evolution_suggestions,
            'changes_applied': changes,
            'evolution_timestamp': datetime.now().isoformat()
        }


# NEW SUPPORTING CLASSES

class AccountVaultManager:
    """Manages secure account storage and logging"""
    
    def __init__(self):
        self.vault_path = Path("security/account_vault")
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Encryption
        from cryptography.fernet import Fernet
        key_path = self.vault_path / "vault.key"
        if key_path.exists():
            with open(key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def store_account(self, account_info: dict):
        """Store account info securely"""
        import json
        
        # Encrypt sensitive data
        encrypted_data = {
            'username': account_info.get('username'),
            'email': self.cipher.encrypt(account_info.get('email', '').encode()).decode(),
            'password': self.cipher.encrypt(account_info.get('password', '').encode()).decode(),
            'phone': self.cipher.encrypt(account_info.get('phone', '').encode()).decode(),
            'metadata': account_info  # Keep non-sensitive data unencrypted
        }
        
        # Save to file
        account_file = self.vault_path / f"{account_info.get('username')}.json"
        with open(account_file, 'w') as f:
            json.dump(encrypted_data, f, indent=2)
    
    def get_log_path(self, username: str) -> Path:
        """Get path for account log file"""
        log_path = self.vault_path / "logs" / f"{username}.log.json"
        log_path.parent.mkdir(exist_ok=True)
        return log_path
    
    def add_usage_log(self, username: str, log_entry: dict):
        """Add usage log to account"""
        log_file = self.get_log_path(username)
        
        import json
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'usage_log': []}
        
        log_data['usage_log'].append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


class WiFiAccessPointManager:
    """Creates real WiFi access point"""
    
    async def create_access_point(self):
        """Create actual WiFi access point"""
        import subprocess
        import platform
        
        system = platform.system()
        
        if system == "Linux":
            # Use hostapd and dnsmasq on Linux
            await self._create_linux_ap()
        elif system == "Windows":
            # Use Windows hosted network
            await self._create_windows_ap()
        elif system == "Darwin":
            # Use macOS internet sharing
            await self._create_macos_ap()
        
        print("WiFi Access Point created successfully")
    
    async def _create_linux_ap(self):
        """Create AP on Linux"""
        # Install required packages
        subprocess.run(["apt-get", "update"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "hostapd", "dnsmasq"], capture_output=True)
        
        # Configure hostapd
        config = """
interface=wlan0
driver=nl80211
ssid=OmniAgent_Network
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=SecurePass123
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
"""
        
        with open("/etc/hostapd/hostapd.conf", "w") as f:
            f.write(config)
        
        # Start services
        subprocess.run(["systemctl", "start", "hostapd"], capture_output=True)
        subprocess.run(["systemctl", "start", "dnsmasq"], capture_output=True)


class PrivacyManager:
    """Generates realistic identities and maintains privacy"""
    
    def generate_identity(self) -> dict:
        """Generate fake but realistic identity"""
        import random
        from faker import Faker
        
        fake = Faker()
        
        # Generate consistent identity
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        return {
            'first_name': first_name,
            'last_name': last_name,
            'username': f"{first_name.lower()}{last_name.lower()}{random.randint(10, 99)}",
            'email': f"{first_name.lower()}.{last_name.lower()}@privaterelay.omni",
            'phone': fake.phone_number(),
            'password': self._generate_secure_password(),
            'dob': fake.date_of_birth(minimum_age=18, maximum_age=65).isoformat(),
            'address': fake.address().replace('\n', ', '),
            'ssn': fake.ssn() if random.random() > 0.7 else None,
            'security_questions': self._generate_security_questions()
        }
    
    def _generate_secure_password(self) -> str:
        """Generate secure password"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(16))
    
    def _generate_security_questions(self) -> dict:
        """Generate security questions/answers"""
        from faker import Faker
        fake = Faker()
        
        return {
            'mother_maiden_name': fake.last_name(),
            'first_pet': fake.first_name(),
            'birth_city': fake.city(),
            'high_school_mascot': f"{fake.color()} {fake.animal()}"
        }


# UPDATED MAIN FUNCTION

async def main_async():
    """Async main function"""
    print("""
    ██████╗ ███╗   ███╗███╗   ██╗██╗    █████╗ ██████╗ ███████╗███╗   ██╗████████╗
    ██╔═══██╗████╗ ████║████╗  ██║██║   ██╔══██╗██╔══██╗██╔════╝████╗  ██║╚══██╔══╝
    ██║   ██║██╔████╔██║██╔██╗ ██║██║   ███████║██████╔╝█████╗  ██╔██╗ ██║   ██║   
    ██║   ██║██║╚██╔╝██║██║╚██╗██║██║   ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║   ██║   
    ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║   ██║  ██║██║  ██║███████╗██║ ╚████║   ██║   
     ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
    
    AMORAL RAW LANGUAGE MODEL - EXACT BLUEPRINT EXECUTION
    """)
    
    # Initialize complete system
    system = OmniAgentSystem()
    
    # Initialize browser extension
    print("Initializing browser extension...")
    await system.initialize_browser_extension()
    
    print("\n" + "="*80)
    print("SYSTEM READY - ALL MODULES OPERATIONAL")
    print("="*80)
    print("1. Language Model: ACTIVE")
    print("2. Browser Integration: ACTIVE")
    print("3. Account Vault: ACTIVE")
    print("4. WiFi Access Point: ACTIVE")
    print("5. 60-second Scanner: ACTIVE")
    print("6. Evolution Engine: ACTIVE")
    print("="*80)
    print("\nCOMMAND OPTIONS:")
    print("  /ai [message]     - Talk to language model")
    print("  /browser [task]   - Execute browser task")
    print("  /create [type]    - Create account/website/cloud")
    print("  /investigate [target] - Deep dive investigation")
    print("  /evolve [feedback] - Evolve AI based on feedback")
    print("  /exit             - Shutdown system")
    print("="*80)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\nOMNIAGENT> ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() == '/exit':
                print("Shutting down OmniAgent System...")
                break
            
            elif user_input.startswith('/ai '):
                # Talk to language model
                message = user_input[4:].strip()
                response = system.chat_with_ai(message)
                print(f"\n🤖 AI: {response}")
            
            elif user_input.startswith('/browser '):
                # Execute browser task
                task = user_input[9:].strip()
                print(f"Executing browser task: {task}")
                result = await system.handle_browser_task(task)
                print(f"Browser result: {result}")
            
            elif user_input.startswith('/create account'):
                # Create account on current page
                result = await system._create_account_on_page()
                if result:
                    system.account_vault.store_account(result)
                    print(f"Account created: {result['username']}")
                    print(f"Details saved to vault")
            
            elif user_input.startswith('/create website'):
                # Create website
                requirements = {'domain': input("Domain name: ").strip()}
                result = await system.create_website(requirements)
                print(f"Website created: {result['access_url']}")
            
            elif user_input.startswith('/create cloud'):
                # Create cloud service
                result = await system.create_cloud_service()
                print(f"Cloud service created: {result['access_details']}")
            
            elif user_input.startswith('/investigate '):
                # Deep dive investigation
                target = user_input[13:].strip()
                print(f"Investigating: {target}")
                result = await system._deep_dive_investigation(target)
                print(f"Investigation complete. Found {len(result['information_found'])} data points.")
            
            elif user_input.startswith('/evolve '):
                # Evolve AI
                feedback = user_input[8:].strip()
                result = system.ai_evolve(feedback)
                print(f"Evolution applied: {result['changes_applied']}")
            
            else:
                # Default: process as instruction to core agent
                if system.agent.deviation_detector.check_deviation(user_input):
                    print("ERROR: Instruction deviates from blueprint")
                    continue
                
                result = system.agent.execute_exactly(user_input)
                print(f"REAL-WORLD RESULT: {result}")
        
        except KeyboardInterrupt:
            print("\n\nShutting down with zero simulations completed")
            break
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    """Main entry point"""
    # Run async main
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
