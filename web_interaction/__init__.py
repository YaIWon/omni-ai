"""
Web Interaction Module - Complete Browser Integration
"""
from .real_browser_integration import RealBrowserMonitor, BrowserAI, BrowserAIContext
from .page_analyzer import PageAnalyzer
from .form_automator import FormAutomator
from .browser_integration import BrowserIntegrationManager

__all__ = [
    'RealBrowserMonitor',
    'BrowserAI', 
    'BrowserAIContext',
    'PageAnalyzer',
    'FormAutomator',
    'BrowserIntegrationManager'
]

# Initialize web interaction components
def initialize_web_interaction():
    """Initialize all web interaction components"""
    from .real_browser_integration import BrowserAI
    return BrowserAI()
