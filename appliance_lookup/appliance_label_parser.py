import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_extracted_field(value, field_name="field"):
    """
    Clean extracted model/serial with context-aware character normalization.
    - If mostly digits: O→0, I→1, S→5, Z→2, B→8
    - If mostly letters: 0→O, 1→I, 5→S, 2→Z, 8→B
    - Special: serials ending in 2 or 0 often should be Q
    - Special: slash in middle of model often should be J
    """
    if not value or len(value) < 2:
        return value
    
    original = value
    
    # Special case for models: slash in middle of alphanumeric string is often J
    # e.g., "CHP9536S/3SS" should be "CHP9536J3SS"
    if field_name == "model" and '/' in value:
        # If slash is surrounded by alphanumerics (not at start/end), likely a J
        if len(value) > 2:
            parts = value.split('/')
            if len(parts) == 2 and parts[0] and parts[1]:
                # Both sides have content - likely misread J
                value = value.replace('/', 'J')
                logging.debug(f"Model slash-to-J fix: '{original}' → '{value}'")
                return value
    
    # Count letters and digits
    letters = sum(c.isalpha() for c in value)
    digits = sum(c.isdigit() for c in value)
    total_alnum = letters + digits
    
    if total_alnum == 0:
        return value
    
    digit_ratio = digits / total_alnum
    
    # Special case for serials: trailing 2 or 0 often should be Q
    if field_name == "serial" and len(value) > 3:
        if value[-1] in ['2', '0'] and value[-2].isdigit():
            # Likely should be Q at the end
            value = value[:-1] + 'Q'
            logging.debug(f"Serial trailing fix: '{original}' → '{value}'")
            return value
    
    # IMPORTANT: Appliance models are often MIXED (like DW80R2031US)
    # Only do aggressive cleanup if VERY heavily weighted one way or the other
    
    # Heavily numeric (>75%) - convert ambiguous chars to digits
    if digit_ratio > 0.75:
        value = value.replace('O', '0')
        value = value.replace('o', '0')
        value = value.replace('I', '1')
        value = value.replace('l', '1')
        value = value.replace('S', '5')
        value = value.replace('Z', '2')
        value = value.replace('z', '2')
        value = value.replace('B', '8')
        if value != original:
            logging.debug(f"Field '{field_name}' cleanup (heavily digits): '{original}' → '{value}'")
    
    # Heavily alphabetic (<25% digits) - convert ambiguous chars to letters
    elif digit_ratio < 0.25:
        value = value.replace('0', 'O')
        value = value.replace('1', 'I')
        value = value.replace('5', 'S')
        value = value.replace('2', 'Z')
        value = value.replace('8', 'B')
        if value != original:
            logging.debug(f"Field '{field_name}' cleanup (heavily letters): '{original}' → '{value}'")
    
    # Mixed alphanumeric (25-75% digits) - DO NOT do blanket conversions!
    # Only fix very obvious contextual errors
    else:
        # Fix leading zero to O if followed by letters (e.g., "0MEGA" -> "OMEGA")
        if len(value) > 1 and value[0] == '0' and value[1].isalpha():
            value = 'O' + value[1:]
            logging.debug(f"Field '{field_name}' cleanup (mixed, leading O fix): '{original}' → '{value}'")
        # Fix lowercase L to 1 when surrounded by digits (e.g., "12l45" -> "12145")
        value = re.sub(r'(\d)[lI](\d)', r'\g<1>1\g<2>', value)
        if value != original:
            logging.debug(f"Field '{field_name}' cleanup (mixed, contextual): '{original}' → '{value}'")
    
    return value

def correct_ocr_errors(text, is_serial=False):
    """
    Correct common OCR character recognition errors
    Context-aware corrections based on whether it's a model or serial number
    """
    if not text or len(text) < 3:
        return text
    
    corrected = text
    
    if is_serial:
        chars = list(corrected)
        for i in range(len(chars)):
            if i < 2:
                if chars[i] == '0':
                    chars[i] = 'O'
                elif chars[i] == '1':
                    chars[i] = 'I'
                elif chars[i] == '8':
                    chars[i] = 'B'
                elif chars[i] == '5':
                    chars[i] = 'S'
                elif chars[i] == 'E' and i == 1:
                    chars[i] = 'B'
            elif i >= 2 and i < len(chars) - 1:
                if chars[i] == 'O':
                    chars[i] = '0'
                elif chars[i] == 'I' or chars[i] == 'l':
                    chars[i] = '1'
                elif chars[i] == 'S':
                    chars[i] = '5'
                elif chars[i] == 'B':
                    chars[i] = '8'
            elif i == len(chars) - 1:
                if chars[i] == '2':
                    chars[i] = 'Q'
                elif chars[i] == '0' and not re.match(r'.*\d$', text[:i]):
                    chars[i] = 'O'
        corrected = ''.join(chars)
    else:
        corrected = corrected.replace('|', 'I')
        corrected = corrected.replace('l', '1')
    
    if corrected != text:
        logging.debug(f"OCR correction: {text} → {corrected}")
    
    return corrected

def parse_appliance_label(ocr_text: str) -> dict:
    """
    Parses appliance OCR text to extract brand, model number, serial number, and other info.
    Specialized for appliance labels with brand-specific pattern recognition.
    """
    logging.debug(f"--- Starting Appliance Label Parse ---")
    logging.debug(f"Raw Text Input:\n{ocr_text}")

    result = {
        "brand": "Unknown",
        "model": "Unknown",
        "serial": "Unknown",
        "other": []
    }

    # Keep original text for reference
    original_text = ocr_text
    
    # Normalize text for easier parsing
    text = ocr_text.upper().replace('\r', ' ')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text_unified = ' '.join(lines)  # For patterns that might span lines
    
    # --- BRAND DETECTION ---
    # Common appliance brands and their identifiers
    brands = {
        'LG': ['LG ELECTRONICS', 'LG APPLIANCE', 'LG.COM', 'LG', '(LG US)', 'LG US', 'LGE.COM'],
        'Samsung': ['SAMSUNG ELECTRONICS', 'SAMSUNG', 'WWW.SAMSUNG.COM'],
        'GE': ['GENERAL ELECTRIC', 'GE APPLIANCES', 'GE PROFILE', 'GE CAFE', 'GE'],
        'Whirlpool': ['WHIRLPOOL', 'WHIRLPOOL CORP', 'WHIRLPOOL CORPORATION', 'WP'],
        'Maytag': ['MAYTAG', 'MAYTAG CORP', 'MAYTAG CORPORATION'],
        'KitchenAid': ['KITCHENAID', 'KITCHEN AID', 'KITCHEN-AID'],
        'Frigidaire': ['FRIGIDAIRE', 'ELECTROLUX HOME PRODUCTS'],
        'Electrolux': ['ELECTROLUX', 'ELECTROLUX HOME PRODUCTS'],
        'Wolf': ['WOLF APPLIANCE', 'WOLF RANGE', 'WOLF'],
        'Viking': ['VIKING RANGE', 'VIKING'],
        'Sub-Zero': ['SUB ZERO', 'SUB-ZERO', 'SUBZERO'],
        'Bosch': ['BOSCH', 'BOSCH HOME'],
        'Kenmore': ['KENMORE', 'SEARS KENMORE'],
        'Amana': ['AMANA']
    }
    
    for brand_name, keywords in brands.items():
        for keyword in keywords:
            if keyword in text:
                result["brand"] = brand_name
                logging.debug(f"Found brand: {brand_name} using keyword '{keyword}'")
                break
        if result["brand"] != "Unknown":
            break
    
    # --- MODEL NUMBER DETECTION ---
    # 1. Direct labeled format (most reliable)
    model_patterns = [
        # General patterns with MODEL keyword - capture until boundary words
        (r"MODEL[\s:#\.-]*NO\.?[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})(?:\s+(?:SERIAL|SER|S/N|TYPE|DATE|MFG|MFD|VOLT|AMPS|WATTS|LISTED|UL|CSA|ETL|$))", "labeled-model-no-bounded"),
        (r"MODEL[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})(?:\s+(?:SERIAL|SER|S/N|TYPE|DATE|MFG|MFD|VOLT|AMPS|WATTS|LISTED|UL|CSA|ETL|$))", "labeled-model-bounded"),
        (r"MOD[\s:#\.-]*NO\.?[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})(?:\s+(?:SERIAL|SER|S/N|TYPE|DATE|MFG|MFD|VOLT|AMPS|WATTS|LISTED|UL|CSA|ETL|$))", "labeled-mod-no-bounded"),
        # Fallback patterns without boundary (but will be cleaned up)
        (r"MODEL[\s:#\.-]*NO\.?[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-model-no"),
        (r"MODEL[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-model"),
        (r"MOD[\s:#\.-]*NO\.?[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-mod-no"),
        (r"MOD[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-mod"),
        (r"TYPE[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-type"),
        (r"CAT\.?[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-cat"),
        (r"CATALOG[\s:#\.-]*([A-Z0-9][A-Z0-9\s/\-]{3,20})", "labeled-catalog"),
        
        # Brand-specific patterns
        # LG models (washing machines, refrigerators, etc)
        (r"\b(WM[0-9]{4}[A-Z]{2,3})\b", "lg-washer"),
        (r"\b(LM[A-Z][A-Z0-9]{5,8})\b", "lg-refrigerator"),
        (r"\b(LDF[0-9]{4}[A-Z]{1,3})\b", "lg-dishwasher"),
        (r"\b(LDE[0-9]{4}[A-Z]{1,3})\b", "lg-dryer"),
        (r"\b(LSE[0-9]{4}[A-Z]{1,3})\b", "lg-range"),
        
        # Samsung models
        (r"\b(RF[0-9]{2}[A-Z][0-9]{4}[A-Z]{2})\b", "samsung-refrigerator"),
        (r"\b(NE[0-9]{2}[A-Z][0-9]{4}[A-Z])\b", "samsung-range"),
        (r"\b(WF[0-9]{2}[A-Z][0-9]{4}[A-Z]{2})\b", "samsung-washer"),
        
        # Frigidaire/Electrolux
        (r"\b(F[FGE][A-Z0-9]{5,10})\b", "frigidaire-model"),
        
        # GE models
        (r"\b(ZI[A-Z]{2}[0-9]{3}[A-Z]{5,6})\b", "ge-zi-series"),  # ZI models like ZICP360NHBRH
        (r"\b(JGB[0-9]{3}[A-Z0-9]{2,5})\b", "ge-range"),
        (r"\b(G[A-Z]{2}[0-9]{3}[A-Z0-9]{2,5})\b", "ge-general"),
        (r"\b(Z[A-Z]{3}[0-9]{3}[A-Z]{2,6})\b", "ge-z-series"),
        
        # Whirlpool models
        (r"\b(WRF[0-9]{3}[A-Z0-9]{3})\b", "whirlpool-refrigerator"),
        (r"\b(WDT[0-9]{3}[A-Z0-9]{3})\b", "whirlpool-dishwasher"),
        
        # Generic model patterns (fallback)
        (r"P/N:?\s*([A-Z0-9\-]{4,15})", "part-number"),
        (r"PN:?\s*([A-Z0-9\-]{4,15})", "part-number"),
        (r"\b([A-Z]{2,4}[0-9]{4,6}[A-Z0-9]{0,3})\b", "generic-alphanumeric"),
    ]
    
    # Try all model patterns
    for pattern, pattern_type in model_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            potential_model = match.group(1).strip().upper()
            
            # Stop at common boundary words if they're in the captured text
            boundary_words = ['SERIAL', 'SER', 'TYPE', 'DATE', 'MFG', 'MFD', 'VOLT', 'AMPS', 'WATTS', 'LISTED', 'UL', 'CSA', 'ETL', 'NUMBER']
            for boundary in boundary_words:
                if boundary in potential_model:
                    potential_model = potential_model.split(boundary)[0].strip()
            
            # Apply context-aware cleanup FIRST (this converts / to J if appropriate)
            potential_model = clean_extracted_field(potential_model, "model")
            # THEN remove any remaining spaces, slashes
            potential_model = re.sub(r'[\s/]+', '', potential_model)
            # Remove trailing underscores or non-alphanumeric junk
            potential_model = re.sub(r'[^A-Z0-9]+$', '', potential_model)
            # Remove leading junk
            potential_model = re.sub(r'^[^A-Z0-9]+', '', potential_model)
            # Filter: must be at least 4 chars and not common junk words - expanded list
            skip_words = ["UL", "CSA", "ETL", "MHZ", "VAC", "HZ", "HOUSEHOLD", "APPLIANCES", "SERIALNO", "SERIAL", 
                         "MODELNO", "MODEL", "MODELE", "NUMERO", "MENTION", "MENTIONTHIS", "WHEN", "CALLING", "SERVICE",
                         "NUMBER", "NO", "MOD", "TYPE", "LISTED", "SN"]
            skip_if_contains = ['MENTION', 'CALLING', 'SERVICE', 'CAUTION', 'WARNING', 'WHEN', 'ORDERING']
            should_skip = potential_model in skip_words or any(phrase in potential_model for phrase in skip_if_contains)
            # Accept reasonable length models (4-25 chars)
            if 4 <= len(potential_model) <= 25 and not should_skip:
                result["model"] = potential_model
                logging.debug(f"Found model: {potential_model} using {pattern_type} pattern")
                break
    
    # If model not found or too short, aggressively look near MODEL keyword
    if result["model"] == "Unknown" or len(result["model"]) < 6:
        # Look for ANY alphanumeric string 8+ characters near "MODEL" keyword, allowing spaces, slashes, hyphens
        for line in lines:
            if "MODEL" in line or "MODELE" in line or "MOD" in line[:10]:  # MOD at start of line
                # Find any alphanumeric strings 8+ characters on this line (allow spaces, slashes, hyphens within)
                candidates = re.findall(r'\b([A-Z0-9]+(?:[\s/\-]*[A-Z0-9]+)*)\b', line)
                for candidate in candidates:
                    # Remove spaces, slashes, hyphens and check length
                    cleaned = re.sub(r'[\s/\-]+', '', candidate)
                    # Apply context-aware cleanup
                    cleaned = clean_extracted_field(cleaned, "model")
                    if len(cleaned) >= 8 and len(cleaned) <= 20:
                        # Skip obviously wrong things
                        if cleaned not in ["HOUSEHOLD", "APPLIANCES", "REFRIGERATOR", "MENAGERE"]:
                            result["model"] = cleaned
                            logging.debug(f"Aggressively grabbed model near MODEL keyword: {result['model']}")
                            break
                if result["model"] != "Unknown" and len(result["model"]) >= 8:
                    break
    
    # If STILL no model, scan entire text for model-like strings (no keyword required)
    if result["model"] == "Unknown" or len(result["model"]) < 6:
        logging.debug("No MODEL keyword found, scanning entire text for model-like strings")
        skip_words = ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE', 'FREEZER',
                      'MENAGERE', 'LISTEOHOUSEHOLD', 'GEAPPLIANCES', 'EOOCEMDOGECAS',
                      'REFAIGERATOR', 'HOUSEROLD', 'OLARGE', 'REFRIGERAHT', 'PRESSURES']
        
        # Collect all candidates and score them
        candidates_with_scores = []
        for line in lines[:15]:  # Focus on first 15 lines
            line_upper = line.strip()
            # Look for 8-14 character alphanumeric strings starting with letter (expanded range)
            candidates = re.findall(r'\b([A-Z][A-Z0-9]{7,14})\b', line_upper)
            for candidate in candidates:
                if candidate in skip_words or candidate.endswith('VAC') or candidate.endswith('HZ'):
                    continue
                # Skip strings that are mostly digits
                digit_count = sum(c.isdigit() for c in candidate)
                if digit_count > len(candidate) * 0.7:
                    continue
                
                # Score candidates
                score = 0
                # Prefer GE patterns
                if result["brand"] == "GE":
                    if candidate.startswith('ZI'):
                        score += 50  # Strong GE ZI-series pattern
                    elif candidate.startswith('Z'):
                        score += 30  # GE Z-series
                    elif candidate.startswith('G'):
                        score += 20  # GE G-series
                # Prefer 10-12 characters
                if 10 <= len(candidate) <= 12:
                    score += 10
                # Prefer balanced mix of letters and numbers
                letter_count = sum(c.isalpha() for c in candidate)
                if 0.3 <= letter_count/len(candidate) <= 0.6:
                    score += 10
                
                candidates_with_scores.append((candidate, score, line))
                logging.debug(f"Candidate: {candidate}, Score: {score}")
        
        # Pick the highest scoring candidate
        if candidates_with_scores:
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            result["model"] = candidates_with_scores[0][0]
            logging.debug(f"Selected best model candidate: {result['model']} with score {candidates_with_scores[0][1]} from line: {candidates_with_scores[0][2]}")
    
    # --- SERIAL NUMBER DETECTION ---
    # 1. Direct labeled format (most reliable)
    serial_patterns = [
        # General patterns with SERIAL keyword
        (r"SERIAL[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-serial"),
        (r"SERIAL NO[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-serial-no"),
        (r"SER[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-ser"),
        (r"S/N[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-s/n"),
        (r"S/NO[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-s/no"),
        (r"SN[\s:#\.-]*([A-Z0-9\-]{5,20})", "labeled-sn"),
        
        # LG serial format
        (r"\b([0-9]{3}[A-Z]{2,3}[0-9]{5,6})\b", "lg-serial"),
        
        # Samsung serial format
        (r"\b([A-Z][0-9]{9,10})\b", "samsung-serial"),
        
        # Generic serial formats (fallback)
        (r"\b([0-9]{8,12})\b", "numeric-serial"),
        (r"\b([A-Z][0-9]{7,11})\b", "alpha-numeric-serial"),
        (r"\b([0-9]{2}[A-Z][0-9]{5,8})\b", "mixed-serial"),
        (r"\b([A-Z]{2}[0-9]{6,10})\b", "alpha-numeric-serial-2"),
    ]
    
    # Try all serial patterns with scoring
    serial_candidates_with_scores = []
    skip_words = ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE', 'FREEZER',
                  'MENAGERE', 'LISTEOHOUSEHOLD', 'GEAPPLIANCES', 'EOOCEMDOGECAS']
    
    for pattern, pattern_type in serial_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            potential_serial = match.group(1).strip().upper()
            # Apply context-aware cleanup (serials are usually mostly digits)
            potential_serial = clean_extracted_field(potential_serial, "serial")
            # Make sure it's not the model we already found
            if potential_serial == result["model"] or potential_serial in skip_words:
                continue
            
            # Score serial candidates
            score = 0
            
            # Bonus for labeled patterns (found with keyword)
            if "labeled" in pattern_type:
                score += 50
            
            # Prefer longer serials (8-15 chars is typical)
            if 8 <= len(potential_serial) <= 15:
                score += 20
            elif len(potential_serial) > 15:
                score -= 10
            
            # Serials are typically more numeric than models
            digit_count = sum(c.isdigit() for c in potential_serial)
            digit_ratio = digit_count / len(potential_serial) if len(potential_serial) > 0 else 0
            
            # Prefer 50-90% digits
            if 0.5 <= digit_ratio <= 0.9:
                score += 30
            elif digit_ratio > 0.9:
                score += 20
            elif digit_ratio < 0.3:
                score -= 10
            
            # LG serial format bonus
            if re.match(r'^\d{3}[A-Z]{2,3}\d{5,6}$', potential_serial):
                score += 50
            
            # GE serial format bonus
            if result["brand"] == "GE" and re.match(r'^[A-Z]{1,2}\d{7,10}$', potential_serial):
                score += 40
            
            # Common format: starts with letters, ends with many digits
            if re.match(r'^[A-Z]{1,3}\d{6,}$', potential_serial):
                score += 25
            
            serial_candidates_with_scores.append((potential_serial, score, pattern_type))
            logging.debug(f"Serial candidate: {potential_serial}, Score: {score}, Type: {pattern_type}")
    
    # Pick the highest scoring candidate
    if serial_candidates_with_scores:
        serial_candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        result["serial"] = serial_candidates_with_scores[0][0]
        logging.debug(f"Selected best serial: {result['serial']} with score {serial_candidates_with_scores[0][1]} using {serial_candidates_with_scores[0][2]}")
    
    # Apply OCR error correction to model and serial
    if result["model"] != "Unknown":
        corrected_model = correct_ocr_errors(result["model"], is_serial=False)
        if corrected_model != result["model"]:
            logging.debug(f"Model corrected: {result['model']} → {corrected_model}")
            result["model"] = corrected_model
    
    if result["serial"] != "Unknown":
        corrected_serial = correct_ocr_errors(result["serial"], is_serial=True)
        if corrected_serial != result["serial"]:
            logging.debug(f"Serial corrected: {result['serial']} → {corrected_serial}")
            result["serial"] = corrected_serial
    
    # --- BRAND-SPECIFIC HANDLING ---
    # If we found a model but no brand, infer the brand from the model
    if result["brand"] == "Unknown" and result["model"] != "Unknown":
        model = result["model"]
        
        # LG patterns
        if model.startswith("WM") or model.startswith("LM") or model.startswith("LD"):
            result["brand"] = "LG"
            logging.debug(f"Inferred brand LG from model {model}")
            
        # Samsung patterns
        elif model.startswith("RF") or model.startswith("NE") or model.startswith("WF"):
            result["brand"] = "Samsung"
            logging.debug(f"Inferred brand Samsung from model {model}")
            
        # Frigidaire patterns
        elif model.startswith("F") and len(model) >= 6:
            if model.startswith("FF") or model.startswith("FG") or model.startswith("FE"):
                result["brand"] = "Frigidaire"
                logging.debug(f"Inferred brand Frigidaire from model {model}")
    
    # --- COLLECT OTHER INFO ---
    # Collect remaining lines as "other"
    for line in lines:
        line_upper = line.upper()
        
        # Skip lines with model or serial
        if result["model"] != "Unknown" and result["model"] in line_upper:
            continue
        if result["serial"] != "Unknown" and result["serial"] in line_upper:
            continue
            
        # Skip lines with common label keywords
        if any(keyword in line_upper for keyword in ["MODEL", "SERIAL", "S/N", "SER", "MOD"]):
            continue
            
        # Add remaining lines to "other"
        result["other"].append(line)
    
    result["other"] = "\n".join(result["other"])
    
    logging.debug(f"--- Parse Results ---")
    logging.debug(f"Brand: {result['brand']}, Model: {result['model']}, Serial: {result['serial']}")
    logging.debug(f"Other: {len(result['other'].splitlines())} lines")
    logging.debug(f"---------------------")
    
    return result