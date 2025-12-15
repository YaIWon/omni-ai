"""
DEEP DIVE INVESTIGATION SYSTEM
"""
import asyncio
import aiohttp
from typing import Dict, List
import json
from datetime import datetime

class DeepDiveInvestigator:
    def __init__(self):
        self.investigations_path = Path("investigations")
        self.investigations_path.mkdir(parents=True, exist_ok=True)
        
    async def investigate(self, target: str) -> Dict:
        """Perform deep dive investigation on target"""
        print(f"Starting deep dive investigation on: {target}")
        
        investigation = {
            'target': target,
            'start_time': datetime.now().isoformat(),
            'target_type': self._determine_target_type(target),
            'sources_checked': [],
            'findings': {}
        }
        
        # Parallel investigations
        tasks = [
            self._investigate_ip(target),
            self._investigate_email(target),
            self._investigate_username(target),
            self._investigate_phone(target),
            self._search_social_media(target),
            self._check_data_breaches(target),
            self._search_public_records(target),
            self._scan_open_ports(target),
            self._check_threat_intelligence(target)
        ]
        
        # Run all investigations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, dict):
                investigation['findings'].update(result)
                investigation['sources_checked'].extend(result.get('sources', []))
        
        investigation['end_time'] = datetime.now().isoformat()
        investigation['total_findings'] = len(investigation['findings'])
        
        # Save investigation
        self._save_investigation(investigation)
        
        return investigation
    
    async def _investigate_ip(self, ip: str) -> Dict:
        """Investigate IP address"""
        if not self._is_valid_ip(ip):
            return {}
        
        findings = {'ip_investigation': {}}
        
        try:
            async with aiohttp.ClientSession() as session:
                # GeoIP lookup
                async with session.get(f'https://ipapi.co/{ip}/json/') as resp:
                    if resp.status == 200:
                        geo_data = await resp.json()
                        findings['ip_investigation']['geo'] = geo_data
                
                # Reverse DNS
                import socket
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                    findings['ip_investigation']['hostname'] = hostname
                except:
                    pass
                
                # WHOIS lookup
                import whois
                try:
                    whois_data = whois.whois(ip)
                    findings['ip_investigation']['whois'] = str(whois_data)
                except:
                    pass
                
                # Threat intelligence
                async with session.get(f'https://api.abuseipdb.com/api/v2/check?ipAddress={ip}') as resp:
                    if resp.status == 200:
                        threat_data = await resp.json()
                        findings['ip_investigation']['threat'] = threat_data
        
        except Exception as e:
            findings['ip_investigation']['error'] = str(e)
        
        return findings
