"""
NATURAL LANGUAGE INTERFACE - TALK TO AI LIKE YOU TALK TO ME
No commands, just conversation. AI understands intent and executes.
"""
import asyncio
import re
from typing import Dict, List, Any
from datetime import datetime

class NaturalLanguageInterface:
    """
    You say: "Create a Gmail account for me"
    AI understands: Browser task -> Account creation -> Execute
    """
    
    def __init__(self, system):
        self.system = system
        self.conversation_history = []
        self.context = {}
        
        # Intent patterns
        self.intent_patterns = {
            'browser': [
                r'(?:go to|navigate to|open|visit|show me).*?(?:http|www\.|\S+\.(com|org|net))',
                r'(?:click|press|select).*?(?:button|link|icon|menu)',
                r'(?:fill|enter|type).*?(?:form|field|input)',
                r'(?:search|find|look up).*?(?:on|in).*?(?:google|website|page)',
                r'(?:scroll|move).*?(?:up|down|left|right)',
                r'(?:take|capture).*?(?:screenshot|picture)',
                r'(?:download|save).*?(?:file|image|video)',
                r'(?:log in|sign in|login).*?(?:to|into)',
                r'(?:create|make).*?(?:account|profile)',
                r'(?:post|share|upload).*?(?:on|to).*?(?:social media|facebook|twitter)'
            ],
            'account_creation': [
                r'(?:create|make|set up).*?(?:account|profile|user)',
                r'(?:sign up|register).*?(?:for|to)',
                r'(?:need|want).*?(?:new).*?(?:email|gmail|yahoo|outlook).*?(?:account)',
                r'(?:get).*?(?:facebook|twitter|instagram).*?(?:account)'
            ],
            'website_creation': [
                r'(?:create|build|make).*?(?:website|site|web page)',
                r'(?:set up|launch).*?(?:online store|blog|portfolio)',
                r'(?:need|want).*?(?:domain|hosting)'
            ],
            'cloud_creation': [
                r'(?:create|set up|provision).*?(?:server|cloud|vm|virtual machine)',
                r'(?:need|want).*?(?:hosting|server space|cloud storage)',
                r'(?:deploy).*?(?:app|application|service).*?(?:to cloud|online)'
            ],
            'investigation': [
                r'(?:investigate|look into|research|find out).*?(?:about|on)',
                r'(?:who is|what is).*?(?:this person|this ip|this email)',
                r'(?:background check|deep dive|search for).*?',
                r'(?:track|trace).*?(?:ip|email|phone number)'
            ],
            'file_operation': [
                r'(?:create|make|write).*?(?:file|document|script)',
                r'(?:edit|modify|change).*?(?:file|code|document)',
                r'(?:delete|remove|erase).*?(?:file|folder)',
                r'(?:move|copy|rename).*?(?:file|folder)',
                r'(?:find|search for).*?(?:file|document)'
            ],
            'system_operation': [
                r'(?:install|setup|configure).*?(?:software|app|program)',
                r'(?:run|execute|start).*?(?:program|script|command)',
                r'(?:check|monitor).*?(?:system|computer|device)',
                r'(?:connect|disconnect).*?(?:wifi|network|internet)'
            ],
            'ai_conversation': [
                r'(?:hi|hello|hey|greetings)',
                r'(?:how are you|what\'?s up|how do you feel)',
                r'(?:thank you|thanks|appreciate it)',
                r'(?:what can you do|help me|show me options)',
                r'(?:explain|tell me about|what is).*?',
                r'(?:why|how|when|where).*?',
                r'^[^?!.]{10,100}\?$'  # General questions
            ]
        }
        
        # Action handlers
        self.action_handlers = {
            'browser': self._handle_browser_intent,
            'account_creation': self._handle_account_creation,
            'website_creation': self._handle_website_creation,
            'cloud_creation': self._handle_cloud_creation,
            'investigation': self._handle_investigation,
            'file_operation': self._handle_file_operation,
            'system_operation': self._handle_system_operation,
            'ai_conversation': self._handle_ai_conversation
        }
    
    async def process_message(self, user_message: str) -> str:
        """
        Process natural language message
        Returns: AI response and executes actions
        """
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Determine intent
        intent = self._determine_intent(user_message)
        
        # Get response from AI
        ai_response = await self._get_ai_response(user_message, intent)
        
        # Execute action based on intent
        action_result = await self._execute_intent_action(intent, user_message)
        
        # Combine response
        final_response = self._format_response(ai_response, action_result)
        
        # Add to history
        self.conversation_history.append({
            'role': 'ai',
            'message': final_response,
            'intent': intent,
            'action_result': action_result,
            'timestamp': datetime.now().isoformat()
        })
        
        return final_response
    
    def _determine_intent(self, message: str) -> str:
        """Determine user intent from message"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return intent
        
        # Default to conversation
        return 'ai_conversation'
    
    async def _get_ai_response(self, message: str, intent: str) -> str:
        """Get conversational response from AI"""
        # Enhance prompt based on intent
        enhanced_prompt = self._enhance_prompt(message, intent)
        
        # Get response from language model
        response = self.system.language_model.process(enhanced_prompt)
        
        return response
    
    def _enhance_prompt(self, message: str, intent: str) -> str:
        """Enhance prompt with context and intent"""
        context = ""
        
        # Add conversation history
        if self.conversation_history:
            recent = self.conversation_history[-3:]  # Last 3 exchanges
            context = "Previous conversation:\n"
            for entry in recent:
                role = "User" if entry['role'] == 'user' else "AI"
                context += f"{role}: {entry['message']}\n"
        
        # Add intent context
        intent_context = {
            'browser': "The user wants to do something in the browser. Provide helpful guidance and confirm you'll execute it.",
            'account_creation': "The user wants to create an account. Explain what you'll do and ask for any specifics.",
            'website_creation': "The user wants to create a website. Ask about type, content, and features.",
            'cloud_creation': "The user wants cloud services. Explain options and ask for requirements.",
            'investigation': "The user wants investigation. Explain what you'll search for and potential findings.",
            'file_operation': "The user wants file operations. Be specific about what you'll do.",
            'system_operation': "The user wants system operations. Explain steps and ask for confirmation.",
            'ai_conversation': "Have a natural conversation. Be helpful and informative."
        }
        
        full_prompt = f"""
        {context}
        
        User Intent: {intent_context.get(intent, 'General conversation')}
        
        Current Message: {message}
        
        Respond naturally as an AI assistant. If executing an action, mention it casually.
        """
        
        return full_prompt
    
    async def _execute_intent_action(self, intent: str, message: str) -> Dict:
        """Execute action based on intent"""
        if intent in self.action_handlers:
            return await self.action_handlers[intent](message)
        return {}
    
    async def _handle_browser_intent(self, message: str) -> Dict:
        """Handle browser-related requests"""
        print(f"Executing browser action: {message}")
        
        # Extract URL if present
        url_match = re.search(r'(https?://\S+|www\.\S+\.\S+)', message)
        if url_match:
            url = url_match.group(0)
            result = await self.system.browser_ai.monitor.navigate_to(url)
            return {'action': 'navigate', 'url': url, 'result': result}
        
        # Extract click action
        if re.search(r'click|press|select', message.lower()):
            # Use AI to determine what to click
            selector = await self._determine_element_selector(message)
            if selector:
                result = await self.system.browser_ai.monitor.click_element(selector)
                return {'action': 'click', 'selector': selector, 'result': result}
        
        # Extract form fill action
        if re.search(r'fill|enter|type', message.lower()):
            form_data = await self._extract_form_data(message)
            if form_data:
                result = await self.system.browser_ai.monitor.fill_form(form_data)
                return {'action': 'fill_form', 'data': form_data, 'result': result}
        
        # Default: analyze current page
        analysis = await self.system.browser_ai.get_page_analysis()
        return {'action': 'page_analysis', 'result': analysis}
    
    async def _handle_account_creation(self, message: str) -> Dict:
        """Handle account creation requests"""
        print(f"Creating account based on: {message}")
        
        # Extract service name
        service = self._extract_service_name(message)
        
        # Generate identity
        identity = self.system.account_creator.create_identity()
        
        # Navigate to service
        urls = {
            'gmail': 'https://accounts.google.com/signup',
            'yahoo': 'https://login.yahoo.com/account/create',
            'outlook': 'https://signup.live.com/',
            'facebook': 'https://www.facebook.com/r.php',
            'twitter': 'https://twitter.com/i/flow/signup',
            'instagram': 'https://www.instagram.com/accounts/emailsignup/'
        }
        
        if service in urls:
            await self.system.browser_ai.monitor.navigate_to(urls[service])
            
            # Wait a bit for page load
            await asyncio.sleep(3)
            
            # Fill registration form
            form_data = self._generate_registration_data(identity, service)
            result = await self.system.browser_ai.monitor.fill_form(form_data, submit=True)
            
            # Store identity
            self.system.account_vault.store_account(identity)
            
            return {
                'action': 'account_creation',
                'service': service,
                'identity': identity['digital']['username'],
                'result': result
            }
        
        return {'action': 'account_creation', 'error': f'Service not supported: {service}'}
    
    async def _handle_website_creation(self, message: str) -> Dict:
        """Handle website creation requests"""
        print(f"Creating website based on: {message}")
        
        # Extract requirements from message
        requirements = self._extract_website_requirements(message)
        
        # Create website
        result = await self.system.create_website(requirements)
        
        return {
            'action': 'website_creation',
            'requirements': requirements,
            'result': result
        }
    
    async def _handle_cloud_creation(self, message: str) -> Dict:
        """Handle cloud service requests"""
        print(f"Creating cloud service based on: {message}")
        
        result = await self.system.create_cloud_service()
        
        return {
            'action': 'cloud_creation',
            'result': result
        }
    
    async def _handle_investigation(self, message: str) -> Dict:
        """Handle investigation requests"""
        print(f"Investigating based on: {message}")
        
        # Extract target
        target = self._extract_target(message)
        
        if target:
            result = await self.system._deep_dive_investigation(target)
            return {
                'action': 'investigation',
                'target': target,
                'result': result
            }
        
        return {'action': 'investigation', 'error': 'No target specified'}
    
    async def _handle_file_operation(self, message: str) -> Dict:
        """Handle file operations"""
        print(f"Performing file operation: {message}")
        
        # Extract file path and operation
        operation = self._extract_file_operation(message)
        
        if operation:
            result = self.system.agent.file_manager.execute_file_operation(operation)
            return {
                'action': 'file_operation',
                'operation': operation,
                'result': result
            }
        
        return {'action': 'file_operation', 'error': 'Could not determine operation'}
    
    async def _handle_system_operation(self, message: str) -> Dict:
        """Handle system operations"""
        print(f"Performing system operation: {message}")
        
        # Execute system command
        result = self.system.agent.real_world_executor.execute_system_command(message)
        
        return {
            'action': 'system_operation',
            'command': message,
            'result': result
        }
    
    async def _handle_ai_conversation(self, message: str) -> Dict:
        """Handle general conversation - no action needed"""
        return {'action': 'conversation', 'executed': False}
    
    # Helper methods
    def _extract_service_name(self, message: str) -> str:
        """Extract service name from message"""
        services = ['gmail', 'yahoo', 'outlook', 'facebook', 'twitter', 'instagram', 
                   'github', 'linkedin', 'reddit', 'discord']
        
        message_lower = message.lower()
        for service in services:
            if service in message_lower:
                return service
        
        return 'generic'
    
    def _extract_target(self, message: str) -> str:
        """Extract investigation target"""
        # Look for IP, email, username patterns
        ip_match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message)
        if ip_match:
            return ip_match.group(0)
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
        if email_match:
            return email_match.group(0)
        
        # Return last word as potential username
        words = message.split()
        return words[-1] if words else ''
    
    def _format_response(self, ai_response: str, action_result: Dict) -> str:
        """Format final response with action results"""
        if not action_result.get('executed', True):
            return ai_response
        
        action = action_result.get('action', '')
        result = action_result.get('result', {})
        
        if action == 'navigate':
            return f"{ai_response}\n\n📡 Navigated to: {result.get('url', 'Unknown')}"
        elif action == 'account_creation':
            return f"{ai_response}\n\n✅ Account created: {result.get('identity', 'Unknown')}"
        elif action == 'website_creation':
            return f"{ai_response}\n\n🌐 Website created at: {result.get('access_url', 'Check logs')}"
        elif action == 'cloud_creation':
            return f"{ai_response}\n\n☁️ Cloud service provisioned"
        elif action == 'investigation':
            return f"{ai_response}\n\n🔍 Investigation complete. Found {result.get('total_findings', 0)} data points."
        
        return ai_response
