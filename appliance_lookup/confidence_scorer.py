import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_confidence_scores(brand, model, serial, raw_text=""):
    """
    Calculate confidence scores (0-100%) for extracted brand, model, and serial.
    
    Scoring criteria:
    - Pattern matching (does it match expected format?)
    - Length validation (appropriate length for the field?)
    - Character composition (right mix of letters/digits?)
    - Brand-specific format validation
    - OCR agreement (multiple OCR engines agree?)
    """
    scores = {
        "brand": 0,
        "model": 0,
        "serial": 0,
        "overall": 0
    }
    
    details = {
        "brand": [],
        "model": [],
        "serial": []
    }
    
    # ===== BRAND CONFIDENCE =====
    brand_score = 0
    known_brands = [
        'GE', 'GENERAL ELECTRIC', 'LG', 'SAMSUNG', 'WHIRLPOOL', 
        'FRIGIDAIRE', 'BOSCH', 'MAYTAG', 'KITCHENAID', 'AMANA',
        'ELECTROLUX', 'KENMORE', 'HAIER', 'SUBZERO', 'VIKING'
    ]
    
    if brand and brand != "Unknown":
        # Known brand: +60 points
        if brand.upper() in known_brands:
            brand_score += 60
            details["brand"].append("Recognized brand name")
        else:
            brand_score += 30
            details["brand"].append("Brand extracted but not in known list")
        
        # Length check: +20 points for reasonable length (2-20 chars)
        if 2 <= len(brand) <= 20:
            brand_score += 20
            details["brand"].append("Reasonable brand name length")
        
        # All caps or title case: +20 points
        if brand.isupper() or brand.istitle():
            brand_score += 20
            details["brand"].append("Proper capitalization")
    else:
        details["brand"].append("Brand not detected")
    
    scores["brand"] = min(brand_score, 100)
    
    # ===== MODEL CONFIDENCE =====
    model_score = 0
    
    if model and model != "Unknown":
        # Length: 6-20 chars is typical: +30 points
        if 6 <= len(model) <= 20:
            model_score += 30
            details["model"].append("Good model length (6-20 chars)")
        elif 4 <= len(model) < 6:
            model_score += 15
            details["model"].append("Short but acceptable model length")
        else:
            details["model"].append("Unusual model length")
        
        # Alphanumeric mix: +25 points
        has_letters = bool(re.search(r'[A-Z]', model))
        has_digits = bool(re.search(r'[0-9]', model))
        if has_letters and has_digits:
            model_score += 25
            details["model"].append("Good alphanumeric mix")
        elif has_letters or has_digits:
            model_score += 10
            details["model"].append("Contains letters or digits")
        
        # Brand-specific format validation
        if brand:
            brand_upper = brand.upper()
            if brand_upper in ['GE', 'GENERAL ELECTRIC']:
                # GE models often start with letters (Z, G, P, C)
                if re.match(r'^[ZGPC]', model):
                    model_score += 20
                    details["model"].append("Matches GE model pattern")
                # Check for valid GE series patterns
                if re.search(r'(Z[A-Z]{2,4}|G[A-Z]{2,4}|P[A-Z]{2,4}|C[A-Z]{2,4})', model):
                    model_score += 15
                    details["model"].append("Valid GE series prefix")
            
            elif brand_upper == 'LG':
                # LG models often have pattern like LT**S, LF**S, LM**S
                if re.match(r'^L[A-Z]\d{2,4}[A-Z]{1,3}', model):
                    model_score += 25
                    details["model"].append("Matches LG model pattern")
            
            elif brand_upper == 'SAMSUNG':
                # Samsung often uses RF, RS, RH prefixes
                if re.match(r'^R[FSHB]\d{2,4}', model):
                    model_score += 25
                    details["model"].append("Matches Samsung model pattern")
            
            elif brand_upper == 'WHIRLPOOL':
                # Whirlpool often uses WR, GI, or model numbers
                if re.match(r'^(WR|GI|WD|WF)', model):
                    model_score += 25
                    details["model"].append("Matches Whirlpool model pattern")
        
        # No weird characters: +15 points
        if re.match(r'^[A-Z0-9-]+$', model):
            model_score += 15
            details["model"].append("Clean format (alphanumeric + hyphens only)")
        
        # Avoid common OCR garbage patterns: -10 points each
        garbage_patterns = ['XXX', '000', '|||', 'III', 'OOO']
        for pattern in garbage_patterns:
            if pattern in model:
                model_score -= 10
                details["model"].append(f"Contains suspicious pattern: {pattern}")
    else:
        details["model"].append("Model not detected")
    
    scores["model"] = max(0, min(model_score, 100))
    
    # ===== SERIAL CONFIDENCE =====
    serial_score = 0
    
    if serial and serial != "Unknown":
        # Length: 8-15 chars is typical: +30 points
        if 8 <= len(serial) <= 15:
            serial_score += 30
            details["serial"].append("Good serial length (8-15 chars)")
        elif 6 <= len(serial) < 8 or 15 < len(serial) <= 18:
            serial_score += 15
            details["serial"].append("Acceptable serial length")
        else:
            details["serial"].append("Unusual serial length")
        
        # Digit percentage: 50-90% is typical: +25 points
        digit_count = sum(c.isdigit() for c in serial)
        digit_ratio = digit_count / len(serial) if len(serial) > 0 else 0
        if 0.5 <= digit_ratio <= 0.9:
            serial_score += 25
            details["serial"].append(f"Good digit ratio ({digit_ratio:.0%})")
        elif 0.3 <= digit_ratio < 0.5 or 0.9 < digit_ratio <= 1.0:
            serial_score += 15
            details["serial"].append(f"Acceptable digit ratio ({digit_ratio:.0%})")
        
        # Brand-specific serial validation
        if brand:
            brand_upper = brand.upper()
            if brand_upper in ['GE', 'GENERAL ELECTRIC']:
                # GE serials often: 2 letters + 6-8 digits
                if re.match(r'^[A-Z]{2}\d{6,8}', serial):
                    serial_score += 20
                    details["serial"].append("Matches GE serial pattern")
            
            elif brand_upper == 'LG':
                # LG serials: 3 digits + 2-3 letters + 5-6 digits
                if re.match(r'^\d{3}[A-Z]{2,3}\d{5,6}', serial):
                    serial_score += 20
                    details["serial"].append("Matches LG serial pattern")
            
            elif brand_upper == 'SAMSUNG':
                # Samsung serials often alphanumeric mix
                if re.match(r'^[A-Z0-9]{10,14}', serial):
                    serial_score += 15
                    details["serial"].append("Matches Samsung serial pattern")
        
        # No weird characters: +15 points
        if re.match(r'^[A-Z0-9]+$', serial):
            serial_score += 15
            details["serial"].append("Clean format (alphanumeric only)")
        
        # Check for sequential patterns (suspicious): -10 points
        if re.search(r'(012|123|234|345|456|567|678|789|ABC|BCD|CDE)', serial):
            serial_score -= 10
            details["serial"].append("Contains sequential pattern (suspicious)")
    else:
        details["serial"].append("Serial not detected")
    
    scores["serial"] = max(0, min(serial_score, 100))
    
    # ===== OVERALL CONFIDENCE =====
    # Weighted average: brand=20%, model=40%, serial=40%
    scores["overall"] = int(
        (scores["brand"] * 0.2) + 
        (scores["model"] * 0.4) + 
        (scores["serial"] * 0.4)
    )
    
    logging.debug(f"Confidence scores: Brand={scores['brand']}%, Model={scores['model']}%, Serial={scores['serial']}%, Overall={scores['overall']}%")
    
    return {
        "scores": scores,
        "details": details,
        "needs_review": scores["overall"] < 70
    }
