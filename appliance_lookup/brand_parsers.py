import re
import logging

"""
Brand-specific parsers for appliance labels.
Each brand has unique model/serial patterns and validation rules.
"""

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class BrandParser:
    """Base class for brand-specific parsers"""
    
    def __init__(self, brand_name):
        self.brand_name = brand_name
    
    def parse(self, text):
        """Override in subclass"""
        return {"model": "Unknown", "serial": "Unknown"}
    
    def validate_model(self, model):
        """Override in subclass for brand-specific validation"""
        return True
    
    def validate_serial(self, serial):
        """Override in subclass for brand-specific validation"""
        return True


class GEParser(BrandParser):
    """GE / General Electric parser"""
    
    def __init__(self):
        super().__init__("GE")
        # GE model patterns
        self.model_patterns = [
            r'\b([ZGPC][A-Z]{2,4}\d{3,4}[A-Z]{2,4})\b',  # Z/G/P/C series
            r'\b(GTS\d{2}[A-Z]{3,4})\b',  # GTS series
            r'\b(GNE\d{2}[A-Z]{3,4})\b',  # GNE series
            r'\b(GSS\d{2}[A-Z]{3,4})\b',  # GSS series
        ]
        # GE serial: typically 2 letters + 6-8 digits
        self.serial_patterns = [
            r'\b([A-Z]{2}\d{6,8})\b',
        ]
    
    def parse(self, text):
        lines = text.upper().split('\n')
        result = {"model": "Unknown", "serial": "Unknown"}
        
        # Find model
        for line in lines:
            if 'MODEL' in line or 'MOD' in line[:15]:
                for pattern in self.model_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["model"] = match.group(1)
                        logging.debug(f"GE parser found model: {result['model']}")
                        break
                if result["model"] != "Unknown":
                    break
        
        # Find serial
        for line in lines:
            if 'SERIAL' in line or 'SER' in line[:15]:
                for pattern in self.serial_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["serial"] = match.group(1)
                        logging.debug(f"GE parser found serial: {result['serial']}")
                        break
                if result["serial"] != "Unknown":
                    break
        
        return result
    
    def validate_model(self, model):
        """Validate GE model format"""
        if len(model) < 6 or len(model) > 15:
            return False
        # Should start with Z, G, P, C, or GTS/GNE/GSS
        if re.match(r'^[ZGPC]', model) or model.startswith(('GTS', 'GNE', 'GSS')):
            return True
        return False
    
    def validate_serial(self, serial):
        """Validate GE serial format"""
        # 2 letters + 6-8 digits
        return bool(re.match(r'^[A-Z]{2}\d{6,8}$', serial))


class LGParser(BrandParser):
    """LG parser"""
    
    def __init__(self):
        super().__init__("LG")
        # LG model patterns: LT**S, LF**S, LM**S, WM****
        self.model_patterns = [
            r'\b(L[TFMRWDGNH]\d{2,4}[A-Z]{1,3})\b',
            r'\b(WM\d{4}[A-Z]{2,4})\b',
            r'\b(LS[DECX]\d{4}[A-Z]{1,2})\b',
        ]
        # LG serial: 3 digits + 2-3 letters + 5-6 digits
        self.serial_patterns = [
            r'\b(\d{3}[A-Z]{2,3}\d{5,6})\b',
            r'\b(\d{3}[A-Z]{2}\d{2}[A-Z]{2}\d{5})\b',
        ]
    
    def parse(self, text):
        lines = text.upper().split('\n')
        result = {"model": "Unknown", "serial": "Unknown"}
        
        # Find model
        for line in lines:
            if 'MODEL' in line or 'MOD' in line[:15]:
                for pattern in self.model_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["model"] = match.group(1)
                        logging.debug(f"LG parser found model: {result['model']}")
                        break
                if result["model"] != "Unknown":
                    break
        
        # Find serial
        for line in lines:
            if 'SERIAL' in line or 'SER' in line[:15]:
                for pattern in self.serial_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["serial"] = match.group(1)
                        logging.debug(f"LG parser found serial: {result['serial']}")
                        break
                if result["serial"] != "Unknown":
                    break
        
        return result
    
    def validate_model(self, model):
        """Validate LG model format"""
        if len(model) < 5 or len(model) > 12:
            return False
        # Should start with L or W
        return model[0] in ('L', 'W')
    
    def validate_serial(self, serial):
        """Validate LG serial format"""
        if len(serial) < 10 or len(serial) > 15:
            return False
        # Should start with 3 digits
        return bool(re.match(r'^\d{3}', serial))


class SamsungParser(BrandParser):
    """Samsung parser"""
    
    def __init__(self):
        super().__init__("Samsung")
        # Samsung model patterns: RF**, RS**, RH**, NE**, DV**
        self.model_patterns = [
            r'\b(R[FSHB]\d{2,4}[A-Z]{2,4})\b',
            r'\b(NE\d{2}[A-Z]{4,6})\b',
            r'\b(DV\d{2}[A-Z]{4,6})\b',
            r'\b(WF\d{2}[A-Z]{4,6})\b',
        ]
        # Samsung serial: alphanumeric mix, 10-14 chars
        self.serial_patterns = [
            r'\b([A-Z0-9]{10,14})\b',
        ]
    
    def parse(self, text):
        lines = text.upper().split('\n')
        result = {"model": "Unknown", "serial": "Unknown"}
        
        # Find model
        for line in lines:
            if 'MODEL' in line or 'MOD' in line[:15]:
                for pattern in self.model_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["model"] = match.group(1)
                        logging.debug(f"Samsung parser found model: {result['model']}")
                        break
                if result["model"] != "Unknown":
                    break
        
        # Find serial
        for line in lines:
            if 'SERIAL' in line or 'SER' in line[:15]:
                for pattern in self.serial_patterns:
                    match = re.search(pattern, line)
                    if match:
                        candidate = match.group(1)
                        # Avoid common non-serial words
                        if candidate not in ['REFRIGERATOR', 'HOUSEHOLD', 'APPLIANCES']:
                            result["serial"] = candidate
                            logging.debug(f"Samsung parser found serial: {result['serial']}")
                            break
                if result["serial"] != "Unknown":
                    break
        
        return result
    
    def validate_model(self, model):
        """Validate Samsung model format"""
        if len(model) < 6 or len(model) > 14:
            return False
        # Should start with common Samsung prefixes
        return model[:2] in ('RF', 'RS', 'RH', 'RB', 'NE', 'DV', 'WF')
    
    def validate_serial(self, serial):
        """Validate Samsung serial format"""
        return 10 <= len(serial) <= 14


class WhirlpoolParser(BrandParser):
    """Whirlpool parser"""
    
    def __init__(self):
        super().__init__("Whirlpool")
        # Whirlpool model patterns: WR**, GI**, WD**, WF**
        self.model_patterns = [
            r'\b(WR[SXFB]\d{1,2}[A-Z]{2,4}\d{1,2})\b',
            r'\b(GI[A-Z]{1,2}\d{2,4}[A-Z]{2,4})\b',
            r'\b(W[DFP]\d{2}[A-Z]{3,5})\b',
        ]
        # Whirlpool serial: alphanumeric, 8-12 chars
        self.serial_patterns = [
            r'\b([A-Z]{2}\d{6,8})\b',
            r'\b([A-Z0-9]{8,12})\b',
        ]
    
    def parse(self, text):
        lines = text.upper().split('\n')
        result = {"model": "Unknown", "serial": "Unknown"}
        
        # Find model
        for line in lines:
            if 'MODEL' in line or 'MOD' in line[:15]:
                for pattern in self.model_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["model"] = match.group(1)
                        logging.debug(f"Whirlpool parser found model: {result['model']}")
                        break
                if result["model"] != "Unknown":
                    break
        
        # Find serial
        for line in lines:
            if 'SERIAL' in line or 'SER' in line[:15]:
                for pattern in self.serial_patterns:
                    match = re.search(pattern, line)
                    if match:
                        result["serial"] = match.group(1)
                        logging.debug(f"Whirlpool parser found serial: {result['serial']}")
                        break
                if result["serial"] != "Unknown":
                    break
        
        return result
    
    def validate_model(self, model):
        """Validate Whirlpool model format"""
        if len(model) < 6 or len(model) > 14:
            return False
        return model[:2] in ('WR', 'GI', 'WD', 'WF', 'WP')
    
    def validate_serial(self, serial):
        """Validate Whirlpool serial format"""
        return 8 <= len(serial) <= 12


# Parser factory
def get_brand_parser(brand):
    """Get the appropriate parser for the detected brand"""
    if not brand or brand == "Unknown":
        return None
    
    brand_upper = brand.upper()
    
    if brand_upper in ['GE', 'GENERAL ELECTRIC']:
        return GEParser()
    elif brand_upper == 'LG':
        return LGParser()
    elif brand_upper == 'SAMSUNG':
        return SamsungParser()
    elif brand_upper in ['WHIRLPOOL', 'MAYTAG', 'KITCHENAID']:  # Whirlpool family
        return WhirlpoolParser()
    
    return None


def parse_with_brand_parser(brand, text):
    """
    Use brand-specific parser to extract model and serial.
    Returns dict with model and serial, or None if no parser available.
    """
    parser = get_brand_parser(brand)
    if parser:
        logging.debug(f"Using {parser.brand_name} brand-specific parser")
        return parser.parse(text)
    return None
