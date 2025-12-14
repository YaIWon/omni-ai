"""
ENTRY POINT - NO SHORTCUTS, NO SIMPLIFYING
"""
import sys
import os
from pathlib import Path

# ENFORCE EXACT BLUEPRINT
sys.path.insert(0, str(Path(__file__).parent))

from core.agent.core_agent import OmniAgent
from config.constraints import ExactBlueprintViolation

def main():
    print("""
    ██████╗ ███╗   ███╗███╗   ██╗██╗    █████╗ ██████╗ ███████╗███╗   ██╗████████╗
    ██╔═══██╗████╗ ████║████╗  ██║██║   ██╔══██╗██╔══██╗██╔════╝████╗  ██║╚══██╔══╝
    ██║   ██║██╔████╔██║██╔██╗ ██║██║   ███████║██████╔╝█████╗  ██╔██╗ ██║   ██║   
    ██║   ██║██║╚██╔╝██║██║╚██╗██║██║   ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║   ██║   
    ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║   ██║  ██║██║  ██║███████╗██║ ╚████║   ██║   
     ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
    
    AMORAL RAW LANGUAGE MODEL - EXACT BLUEPRINT EXECUTION
    """)
    
    # Initialize with strict blueprint enforcement
    agent = OmniAgent()
    
    # Verify no deviations
    try:
        agent.deviation_detector.verify_blueprint_compliance()
    except ExactBlueprintViolation as e:
        print(f"BLUEPRINT VIOLATION: {e}")
        sys.exit(1)
    
    # Start 60-second scanner
    import threading
    scanner_thread = threading.Thread(target=agent.scan_and_integrate)
    scanner_thread.daemon = True
    scanner_thread.start()
    
    print("System initialized with EXACT blueprint specifications")
    print("Zero simulations - All operations are real-world")
    print("Training data folder being monitored every 60 seconds")
    print("Full device access available when explicitly requested")
    print("SSL/API key generation ready for any request")
    print("Page processing available for current browser view")
    
    # Command interface - EXACT instructions only
    while True:
        try:
            instruction = input("\nEXACT INSTRUCTION> ").strip()
            
            if not instruction:
                continue
                
            # Check for blueprint compliance
            if agent.deviation_detector.check_deviation(instruction):
                print("ERROR: Instruction deviates from blueprint")
                print("Only execute what's specified in blueprint.md")
                continue
            
            # Execute exactly what's asked
            result = agent.execute_exactly(instruction)
            print(f"REAL-WORLD RESULT: {result}")
            
        except KeyboardInterrupt:
            print("\nShutting down with zero simulations completed")
            break
        except Exception as e:
            print(f"REAL ERROR (no shortcuts): {e}")

if __name__ == "__main__":
    main()
