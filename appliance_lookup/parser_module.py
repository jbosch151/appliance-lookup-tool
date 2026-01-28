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
    
    # Common OCR mistakes - apply based on context
    if is_serial:
        # Serial numbers typically have pattern like: AB01234567 (2 letters then digits)
        # First 1-2 characters are usually letters
        # Position 2+ are usually digits (except possibly last char)
        
        chars = list(corrected)
        for i in range(len(chars)):
            # First 1-2 characters are usually letters in serials
            if i < 2:
                if chars[i] == '0':
                    chars[i] = 'O'  # 0 → O in letter positions
                elif chars[i] == '1':
                    chars[i] = 'I'  # 1 → I in letter positions
                elif chars[i] == '8':
                    chars[i] = 'B'  # 8 → B in letter positions
                elif chars[i] == '5':
                    chars[i] = 'S'  # 5 → S in letter positions
                elif chars[i] == 'E' and i == 1:  # Second position E is often B
                    chars[i] = 'B'
            # Position 2 onwards (except last) are usually digits
            elif i >= 2 and i < len(chars) - 1:
                if chars[i] == 'O':
                    chars[i] = '0'  # O → 0 in digit positions
                elif chars[i] == 'I' or chars[i] == 'l':
                    chars[i] = '1'  # I/l → 1 in digit positions
                elif chars[i] == 'S':
                    chars[i] = '5'  # S → 5 in digit positions
                elif chars[i] == 'B':
                    chars[i] = '8'  # B → 8 in digit positions
            # Last character - could be a letter (like Q, A, etc.)
            elif i == len(chars) - 1:
                if chars[i] == '2':
                    chars[i] = 'Q'  # 2 → Q at end of serial
                elif chars[i] == '0' and not re.match(r'.*\d$', text[:i]):
                    # Only convert if it looks like it should be a letter
                    chars[i] = 'O'  # 0 → O at end if pattern suggests letter
        
        corrected = ''.join(chars)
    else:
        # Model numbers have mixed patterns - be more conservative
        # Common fixes that apply broadly
        corrected = corrected.replace('|', 'I')  # | → I
        corrected = corrected.replace('l', '1')  # lowercase l → 1 in model numbers
    
    if corrected != text:
        logging.debug(f"OCR correction: {text} → {corrected}")
    
    return corrected

def parse_appliance_text(text):
    """
    Improved parser with enhanced pattern matching to correctly identify
    model and serial numbers even when not explicitly labeled
    """
    logging.debug(f"--- Starting Enhanced Parse ---")
    logging.debug(f"Raw Text Input:\n{text}")

    brand = "Unknown"
    model = "Unknown"
    serial = "Unknown"
    other_lines = []
    model_candidates = []
    serial_candidates = []

    # Clean up text - remove extra whitespace, split into lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text_lower = text.lower()
    
    # Brand recognition
    brands = {
        'LG': ['lg electronics', 'lg appliance', 'lg.com', 'lg'],
        'Samsung': ['samsung electronics', 'samsung'],
        'GE': ['general electric', 'ge appliances', 'ge profile', 'ge cafe', 'ge'],
        'Whirlpool': ['whirlpool', 'whirlpool corp'],
        'Maytag': ['maytag', 'maytag corp'],
        'KitchenAid': ['kitchenaid', 'kitchen aid'],
        'Frigidaire': ['frigidaire'],
        'Electrolux': ['electrolux home products', 'electrolux'],
        'Wolf': ['wolf appliance', 'wolf range', 'wolf'],
        'Viking': ['viking range', 'viking'],
        'Sub-Zero': ['sub zero', 'sub-zero', 'subzero'],
        'Bosch': ['bosch', 'bosch home'],
        'Kenmore': ['kenmore', 'sears kenmore'],
        'Amana': ['amana'],
        'Dacor': ['dacor'],
        'Fisher & Paykel': ['fisher & paykel', 'fisher and paykel'],
        'Miele': ['miele'],
        'Thermador': ['thermador']
    }
    
    # Find brand by direct text matching
    for brand_name, keywords in brands.items():
        for keyword in keywords:
            if keyword in text_lower:
                brand = brand_name
                logging.debug(f"Found brand: {brand} using keyword '{keyword}'")
                break
        if brand != "Unknown":
            break
    
    # Define patterns for direct extraction with labels
    model_keyword_patterns = [
        r'\b(?:model|mod|model no|model#|m#|mod#|type|cat|catalog)[\s:=#-]*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+)*)',
        r'\bmodel[\s:]*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',  # Up to 5 segments
        r'\bmod[\s:]*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'\bm#[\s:]*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'\btype[\s:]*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'p/n:?\s*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'pn:?\s*([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'model:([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})',
        r'model\s+([a-zA-Z0-9-]+(?:[\s/\-]*[a-zA-Z0-9]+){0,5})'
    ]
    
    serial_keyword_patterns = [
        r'\b(?:serial|ser|s/n|serial no|serial#|ser#|sn)[\s:=#-]*([a-zA-Z0-9-]{5,15})\b',
        r'\bserial[\s:]*([a-zA-Z0-9-]{5,15})\b',
        r'\bser[\s:]*([a-zA-Z0-9-]{5,15})\b',
        r'\bs/n[\s:]*([a-zA-Z0-9-]{5,15})\b',
        r'\bsn[\s:]*([a-zA-Z0-9-]{5,15})\b',
        # Added patterns for "S/No" which is common on LG appliances
        r's/no\.?:?\s*([a-zA-Z0-9-]{5,15})',
        r's/no\.?\s*([a-zA-Z0-9-]{5,15})',
        r's/no([a-zA-Z0-9-]{5,15})'
    ]
    
    # Brand-specific model patterns (without labels)
    brand_model_patterns = {
        'Frigidaire': [r'\b(F[FGE][A-Z0-9]{5,10})\b'],
        'LG': [
            r'\b(L[GDTWS][A-Z0-9]{5,10})\b',
            r'\b(WM[0-9]{4}[A-Z]{2,3})\b',  # LG washing machine models like WM3670HWA
            r'\b(DLE[0-9]{4}[A-Z]{1,3})\b', # LG dryer models
            r'\b(LDF[0-9]{4}[A-Z]{1,3})\b', # LG dishwasher models
            r'\b(LMXS[0-9]{2}[A-Z]{3})\b',  # LG refrigerator models
            r'\b(LRE[0-9]{4}[A-Z]{1,3})\b'  # LG range models
        ],
        'Samsung': [r'\b(ME[0-9]{2}[A-Z][0-9]{4}[A-Z]{2})\b', r'\b(NE[0-9]{2}[A-Z][0-9]{4}[A-Z])\b'],
        'Whirlpool': [r'\b(WP[A-Z0-9]{7,10})\b', r'\b(W[DFT][0-9]{2}[A-Z0-9]{4,6})\b'],
        'GE': [
            r'\b(ZI[A-Z]{2}[0-9]{3}[A-Z]{5,6})\b',  # GE models like ZICP360NHBRH, ZICRSEORTORH
            r'\b(JGB[0-9]{3}[A-Z0-9]{2,5})\b', 
            r'\b(PGB[0-9]{3}[A-Z0-9]{2,5})\b',
            r'\b(G[A-Z]{2}[0-9]{3}[A-Z0-9]{2,5})\b',  # GE general pattern
            r'\b(Z[A-Z]{3}[0-9]{3}[A-Z]{2,6})\b'  # Z-series models
        ],
        'KitchenAid': [r'\b(KDTE[A-Z0-9]{3,7})\b', r'\b(KDPE[A-Z0-9]{3,7})\b'],
        'Maytag': [r'\b(MDB[A-Z0-9]{4,8})\b', r'\b(MMV[A-Z0-9]{4,8})\b']
    }
    
    # Common model formats (fallback for unlabeled models)
    common_model_patterns = [
        r'\b(WM[0-9]{4}[A-Z]{1,3})\b',       # Common LG washer pattern
        r'\b([A-Z]{2,4}[0-9]{4,6}[A-Z0-9]{0,3})\b',  # Matches patterns like FFMV1645TD
        r'\b([A-Z]{1,3}-?[0-9]{3,6}[A-Z]{0,2})\b'    # Matches patterns like GX-9855A
    ]
    
    # Common serial formats (fallback for unlabeled serials)
    common_serial_patterns = [
        # LG common serial patterns
        r'\b([0-9]{3}[A-Z]{2,3}[0-9]{5,6})\b',    # LG format like 707TWGG04983
        r'\b([0-9]{3}[A-Z]{2}[0-9]{5,6})\b',      # LG format variant
        # Other common patterns
        r'\b([0-9]{8,12})\b',                     # Pure numeric: 123456789012
        r'\b([A-Z][0-9]{7,11})\b',                # Alpha-numeric: A12345678901
        r'\b([0-9]{2}[A-Z][0-9]{5,8})\b',         # Mixed: 12A1234567
        r'\b([A-Z]{2}[0-9]{6,10})\b',             # Alpha-numeric: AB1234567890
        r'\b([A-Z0-9]{7,15})\b',                  # General long alphanumeric string
        r'\b([A-Z]{1,2}[0-9]{5,12})\b',           # Letter(s) followed by numbers
        r'\b([0-9]{5,10}[A-Z]{1,2})\b'            # Numbers followed by letter(s)
    ]

    # Step 1: Check for model with explicit keywords
    for line in lines:
        for pattern in model_keyword_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                potential_model = match.group(1).upper().strip()
                
                # Stop at common boundary words if they're in the captured text
                boundary_words = ['SERIAL', 'SER', 'TYPE', 'DATE', 'MFG', 'MFD', 'VOLT', 'AMPS', 'WATTS', 'LISTED', 'UL', 'CSA', 'ETL', 'NUMBER', 'NO']
                for boundary in boundary_words:
                    if boundary in potential_model:
                        potential_model = potential_model.split(boundary)[0].strip()
                
                # Apply context-aware cleanup FIRST (this converts / to J if appropriate)
                potential_model = clean_extracted_field(potential_model, "model")
                
                # THEN remove any remaining spaces, slashes
                potential_model = re.sub(r'[\s/]+', '', potential_model)
                # Remove leading/trailing non-alphanumeric
                potential_model = re.sub(r'^[^A-Z0-9]+', '', potential_model)
                potential_model = re.sub(r'[^A-Z0-9]+$', '', potential_model)
                
                # IMPORTANT: Check next line too - OCR sometimes splits model numbers
                line_idx = lines.index(line)
                if line_idx + 1 < len(lines):
                    next_line = lines[line_idx + 1].strip()
                    # If next line starts with slash or is short alphanumeric, it might be part of model
                    if next_line and (next_line.startswith('/') or (len(next_line) <= 6 and re.match(r'^[A-Z0-9]+$', next_line))):
                        append_text = re.sub(r'[\s/]+', '', next_line)
                        potential_model += append_text
                        app.logger.debug(f"Appended next line to model: {append_text}")
                # Filter out label text and junk - expanded list
                skip_words = ['HOUSEHOLD', 'APPLIANCES', 'SERIALNO', 'SERIAL', 'MODELNO', 'MODEL', 'MODELE', 'NUMERO',
                              'MENTION', 'MENTIONTHIS', 'WHEN', 'CALLING', 'SERVICE', 'CAUTION', 'WARNING', 'NUMBER', 'NO', 'MOD', 'TYPE', 'LISTED', 'SN']
                # Also skip if it contains common label phrases
                skip_if_contains = ['MENTION', 'CALLING', 'SERVICE', 'CAUTION', 'WARNING', 'WHEN', 'ORDERING']
                should_skip = potential_model in skip_words or any(phrase in potential_model for phrase in skip_if_contains)
                # Accept if it's at least 4 characters, looks like a model, and not too long (< 25 chars)
                if 4 <= len(potential_model) <= 25 and not should_skip:
                    model = potential_model
                    logging.debug(f"Found model with keyword: {model} in line: {line}")
                    break
        if model != "Unknown":
            break
    
    # Step 1b: If model not found or value is short/garbage, aggressively look for any long string near "model"
    if model == "Unknown" or len(model) < 6:
        # Try to find ANY alphanumeric string 8+ chars near model keyword, allowing spaces, slashes, hyphens
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if 'MODEL' in line_upper or 'MOD' in line_upper[:15]:
                # Look for any 8-20 character alphanumeric strings (allow spaces, slashes, hyphens within)
                candidates = re.findall(r'\b([A-Z0-9]+(?:[\s/\-]*[A-Z0-9]+)*)\b', line_upper)
                for candidate in candidates:
                    # Remove spaces, slashes, hyphens and check length
                    cleaned = re.sub(r'[\s/\-]+', '', candidate)
                    # Apply context-aware cleanup
                    cleaned = clean_extracted_field(cleaned, "model")
                    # Accept longer models (8-25 chars) and check against skip list
                    if 8 <= len(cleaned) <= 25:
                        # Skip obvious non-models
                        if cleaned not in ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE']:
                            model = cleaned
                            logging.debug(f"Aggressively grabbed model near keyword: {model}")
                            break
                if model != "Unknown" and len(model) >= 8:
                    break
                # Also check next line
                if i + 1 < len(lines):
                    candidates = re.findall(r'\b([A-Z0-9]+(?:[\s/\-]*[A-Z0-9]+)*)\b', lines[i + 1].upper())
                    for candidate in candidates:
                        cleaned = re.sub(r'[\s/\-]+', '', candidate)
                        # Apply context-aware cleanup
                        cleaned = clean_extracted_field(cleaned, "model")
                        if len(cleaned) >= 8 and len(cleaned) <= 20:
                            if cleaned not in ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE']:
                                model = cleaned
                            logging.debug(f"Aggressively grabbed model from next line: {model}")
                            break
                if model != "Unknown" and len(model) >= 8:
                    break
    
    # Step 1c: If still no model, scan ENTIRE text for GE-format model numbers (10-14 chars alphanumeric)
    if model == "Unknown" or len(model) < 6:
        logging.debug("No MODEL keyword found, scanning entire text for model-like strings")
        skip_words = ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE', 'FREEZER', 
                      'MENAGERE', 'LISTEOHOUSEHOLD', 'GEAPPLIANCES', 'EOOCEMDOGECAS',
                      'REFAIGERATOR', 'HOUSEROLD', 'OLARGE', 'REFRIGERAHT', 'PRESSURES']
        
        # Collect all candidates and score them
        candidates_with_scores = []
        for line in lines[:15]:  # Focus on first 15 lines where model usually appears
            line_upper = line.upper().strip()
            # Look for alphanumeric strings 8-14 characters (expanded range for models like CHP9536J3SS)
            candidates = re.findall(r'\b([A-Z][A-Z0-9]{7,14})\b', line_upper)
            for candidate in candidates:
                if candidate in skip_words or candidate.endswith('VAC') or candidate.endswith('HZ'):
                    continue
                # Skip strings that are mostly digits (likely serial numbers)
                digit_count = sum(c.isdigit() for c in candidate)
                if digit_count > len(candidate) * 0.7:  # More than 70% digits
                    continue
                
                # Score candidates (higher is better)
                score = 0
                # Prefer GE patterns: Z or G followed by letters then numbers
                if brand == "GE":
                    if candidate.startswith('ZI'):
                        score += 50  # Strong GE pattern
                    elif candidate.startswith('Z'):
                        score += 30  # GE Z-series
                    elif candidate.startswith('G'):
                        score += 20  # GE G-series
                # Prefer 10-12 characters (typical appliance model length)
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
            model = candidates_with_scores[0][0]
            logging.debug(f"Selected best model candidate: {model} with score {candidates_with_scores[0][1]} from line: {candidates_with_scores[0][2]}")

    # Step 2: Check for serial with explicit keywords
    for line in lines:
        for pattern in serial_keyword_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                serial = match.group(1).upper()
                # Apply context-aware cleanup (serials are usually mostly digits)
                serial = clean_extracted_field(serial, "serial")
                logging.debug(f"Found serial with keyword: {serial} in line: {line}")
                break
        if serial != "Unknown":
            break
    
    # Step 3: If we know the brand, check brand-specific model patterns
    if model == "Unknown" and brand in brand_model_patterns:
        for pattern in brand_model_patterns[brand]:
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    model = match.group(1).upper()
                    logging.debug(f"Found {brand} model with pattern: {model} in line: {line}")
                    break
            if model != "Unknown":
                break
    
    # Step 4: Look for unlabeled model numbers using common patterns
    if model == "Unknown":
        for line in lines:
            for pattern in common_model_patterns:
                matches = re.findall(pattern, line.upper())
                for match in matches:
                    # Skip if it looks like a date or short code
                    if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', match) or len(match) < 5:
                        continue
                    # Skip common non-model tokens
                    if match in ["UL", "FCC", "CSA", "ETL", "MHZ", "VAC", "HZ"]:
                        continue
                    model_candidates.append((match, line))
        
        if model_candidates:
            # Sort by likelihood (prefer longer matches)
            model_candidates.sort(key=lambda x: len(x[0]), reverse=True)
            model = model_candidates[0][0]
            logging.debug(f"Found model using common pattern: {model} from line: {model_candidates[0][1]}")
    
    # Step 5: Look for unlabeled serial numbers using common patterns with scoring
    if serial == "Unknown":
        serial_candidates_with_scores = []
        skip_words = ['HOUSEHOLD', 'APPLIANCES', 'REFRIGERATOR', 'APPLIANCE', 'FREEZER',
                      'MENAGERE', 'LISTEOHOUSEHOLD', 'GEAPPLIANCES', 'EOOCEMDOGECAS']
        
        for line in lines:
            for pattern in common_serial_patterns:
                matches = re.findall(pattern, line.upper())
                for match in matches:
                    # Make sure it's not the model number we already found
                    if match == model or match in skip_words:
                        continue
                    
                    # Score serial candidates
                    score = 0
                    
                    # Prefer longer serials (8-15 chars is typical)
                    if 8 <= len(match) <= 15:
                        score += 20
                    elif len(match) > 15:
                        score -= 10  # Too long, likely not a serial
                    
                    # Serials are typically more numeric than models
                    digit_count = sum(c.isdigit() for c in match)
                    digit_ratio = digit_count / len(match) if len(match) > 0 else 0
                    
                    # Prefer 50-90% digits (typical serial pattern)
                    if 0.5 <= digit_ratio <= 0.9:
                        score += 30
                    elif digit_ratio > 0.9:
                        score += 20  # All digits is okay too
                    elif digit_ratio < 0.3:
                        score -= 10  # Too few digits for a serial
                    
                    # LG serial format: 3 digits + 2-3 letters + 5-6 digits
                    if re.match(r'^\d{3}[A-Z]{2,3}\d{5,6}$', match):
                        score += 50
                    
                    # GE serial format: letter + lots of digits
                    if brand == "GE" and re.match(r'^[A-Z]{1,2}\d{7,10}$', match):
                        score += 40
                    
                    # Common format: starts with letters, ends with many digits
                    if re.match(r'^[A-Z]{1,3}\d{6,}$', match):
                        score += 25
                    
                    # Avoid strings that look like dates or model numbers
                    if re.match(r'^\d{6,8}$', match):  # Pure numeric 6-8 digits might be date
                        score -= 5
                    
                    serial_candidates_with_scores.append((match, score, line))
                    logging.debug(f"Serial candidate: {match}, Score: {score}")
        
        if serial_candidates_with_scores:
            # Sort by score (highest first)
            serial_candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
            serial = serial_candidates_with_scores[0][0]
            logging.debug(f"Selected best serial candidate: {serial} with score {serial_candidates_with_scores[0][1]} from line: {serial_candidates_with_scores[0][2]}")
    
    # Apply OCR error correction to model and serial
    if model != "Unknown":
        corrected_model = correct_ocr_errors(model, is_serial=False)
        if corrected_model != model:
            logging.debug(f"Model corrected: {model} → {corrected_model}")
            model = corrected_model
    
    if serial != "Unknown":
        corrected_serial = correct_ocr_errors(serial, is_serial=True)
        if corrected_serial != serial:
            logging.debug(f"Serial corrected: {serial} → {corrected_serial}")
            serial = corrected_serial
    
    # Step 6: Infer brand from model number if brand is unknown
    if brand == "Unknown" and model != "Unknown":
        # LG models - checking first since they can have several patterns
        if (model.startswith('L') or 
            re.match(r'^[0-9]{3}[A-Z]{2}[0-9]{2,}', model) or  # LG often uses numbers followed by letters format
            model.startswith('MC') or 
            model.startswith('MS')):
            
            brand = "LG"
            logging.debug(f"Inferred brand LG from model {model}")
        
        # Frigidaire/Electrolux models often start with F
        elif model.startswith('F') and len(model) >= 6:
            if model.startswith('FF') or model.startswith('FG') or model.startswith('FE'):
                brand = "Frigidaire"
                logging.debug(f"Inferred brand Frigidaire from model {model}")
                
        # Samsung models often have specific formats
        elif (model.startswith('ME') or 
              model.startswith('NE') or 
              model.startswith('SM') or 
              model.startswith('RF') or
              model.startswith('UN')):
            brand = "Samsung"
            logging.debug(f"Inferred brand Samsung from model {model}")
            
        # Whirlpool models often start with W
        elif model.startswith('W') and len(model) >= 5:
            if model.startswith('WP') or model.startswith('WDF') or model.startswith('WTW'):
                brand = "Whirlpool"
                logging.debug(f"Inferred brand Whirlpool from model {model}")
                
        # GE models often start with G or specific prefixes
        elif (model.startswith('G') or model.startswith('JGB') or 
              model.startswith('PGB') or model.startswith('JB')):
            brand = "GE"
            logging.debug(f"Inferred brand GE from model {model}")
            
        # KitchenAid models often start with K
        elif model.startswith('K') and len(model) >= 5:
            if model.startswith('KDTE') or model.startswith('KDPE') or model.startswith('KSM'):
                brand = "KitchenAid"
                logging.debug(f"Inferred brand KitchenAid from model {model}")
    
    # Step 7: Check for contextual brand clues
    if brand == "Unknown":
        # Look for keywords that often appear with specific brands
        context_brand_clues = {
            "LG": ["south korea", "seoul", "changwon", "lg.com", "lge.com", "life's good", 
                  "life is good", "www.lg.com", "korea", "smart inverter", "inverter linear"],
            "Whirlpool": ["benton harbor", "michigan", "amana", "maytag"],
            "GE": ["general electric", "louisville", "ge.com"],
            "Frigidaire": ["electrolux", "charlotte", "north carolina", "nc"],
            "Samsung": ["suwon", "korea", "samsung.com", "digital inverter"],
            "Bosch": ["bsh", "germany", "munich"],
            "Electrolux": ["charlotte", "stockholm", "sweden"],
            "KitchenAid": ["benton harbor", "michigan"]
        }
        
        for b_name, clues in context_brand_clues.items():
            for clue in clues:
                if clue in text_lower:
                    brand = b_name
                    logging.debug(f"Inferred brand {brand} from contextual clue '{clue}'")
                    break
            if brand != "Unknown":
                break

    # Step 8: Look for known model prefixes in the entire text
    if brand == "Unknown":
        # LG often has these prefixes anywhere in the text
        lg_prefixes = ["LMXS", "LMXC", "LMV", "LMC", "LSE", "LTCS", "LRFX", "LRFV", "LREL", "LRGL", "LRG", "LDF", "LDP"]
        for prefix in lg_prefixes:
            if prefix in text.upper():
                brand = "LG"
                logging.debug(f"Identified LG brand from prefix {prefix} in text")
                break
    
    # Build the "other" section with remaining content
    for line in lines:
        # Skip lines that contain identified model/serial numbers
        if model != "Unknown" and model in line.upper():
            continue
        if serial != "Unknown" and serial in line.upper():
            continue
        
        # Skip lines explicitly containing model/serial keywords
        contains_keyword = False
        for kw in ['model', 'serial', 'mod', 'ser', 's/n']:
            if kw in line.lower():
                contains_keyword = True
                break
        
        if not contains_keyword:
            other_lines.append(line)
    
    other_info = "\n".join(other_lines)

    logging.debug(f"--- Parse Results ---")
    logging.debug(f"Brand: {brand}, Model: {model}, Serial: {serial}")
    logging.debug(f"Other: {len(other_lines)} lines")
    logging.debug(f"---------------------")

    return {
        "brand": brand,
        "model": model,
        "serial": serial,
        "other": other_info
    }
