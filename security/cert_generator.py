"""
SSL CERTIFICATE GENERATOR - ANY KIND
"""
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import datetime
import ipaddress

class CertGenerator:
    def create_any_cert(self, 
                       cert_type: str, 
                       domains: list, 
                       ips: list = None,
                       validity_days: int = 365,
                       **kwargs):
        """Create ANY SSL certificate as requested"""
        
        # Generate private key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=kwargs.get('key_size', 2048),
            backend=default_backend()
        )
        
        # Subject - exact as requested
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, kwargs.get('country', 'US')),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, 
                             kwargs.get('state', 'California')),
            x509.NameAttribute(NameOID.LOCALITY_NAME, 
                             kwargs.get('locality', 'San Francisco')),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, 
                             kwargs.get('org', 'OmniAgent')),
            x509.NameAttribute(NameOID.COMMON_NAME, 
                             kwargs.get('common_name', domains[0]))
        ])
        
        # Build certificate based on exact type
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(subject)  # Self-signed unless specified
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        )
        
        # Add ALL requested domains
        san = x509.SubjectAlternativeName([
            x509.DNSName(domain) for domain in domains
        ])
        
        # Add IP addresses if requested
        if ips:
            for ip in ips:
                san._add_entry(x509.IPAddress(ipaddress.ip_address(ip)))
        
        builder = builder.add_extension(san, critical=False)
        
        # Certificate type specific extensions
        if cert_type.lower() == 'wildcard':
            builder = builder.add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True
            )
        elif cert_type.lower() == 'ev':
            # Extended Validation certificate
            builder = builder.add_extension(
                x509.CertificatePolicies([
                    x509.PolicyInformation(
                        x509.ObjectIdentifier('2.23.140.1.1'),
                        None
                    )
                ]),
                critical=False
            )
        
        # Sign certificate
        certificate = builder.sign(
            private_key=key,
            algorithm=hashes.SHA256(),
            backend=default_backend()
        )
        
        return {
            'certificate': certificate.public_bytes(
                encoding=serialization.Encoding.PEM
            ),
            'private_key': key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        }
