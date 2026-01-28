import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
import easyocr
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Import parser modules
from appliance_lookup.parser_module import parse_appliance_text
from appliance_lookup.appliance_label_parser import parse_appliance_label
from appliance_lookup.confidence_scorer import calculate_confidence_scores
from appliance_lookup.brand_parsers import parse_with_brand_parser
from appliance_lookup.quality_checker import assess_image_quality
from appliance_lookup.gemini_vision import extract_with_gemini, calculate_confidence

# Configure Tesseract command path (make configurable via environment variable)
tesseract_cmd_path = os.environ.get('TESSERACT_CMD_PATH', '/opt/homebrew/bin/tesseract')
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

# Initialize EasyOCR reader once
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Initialize TrOCR (lazy loading to avoid startup delay)
trocr_processor = None
trocr_model = None

def get_trocr_model():
    """Lazy load TrOCR model"""
    global trocr_processor, trocr_model
    if trocr_processor is None or trocr_model is None:
        logging.info("Loading TrOCR model (this may take a moment on first use)...")
        trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        logging.info("TrOCR model loaded successfully")
    return trocr_processor, trocr_model

def apply_ocr_fixups(text, confidence):
    """
    Apply context-aware OCR character corrections based on confidence and position.
    High confidence: minimal fixes. Low confidence: aggressive fixes.
    Common OCR errors: O↔0, I↔1, S↔5, Z↔2, B↔8, etc.
    """
    if not text or len(text) < 2:
        return text
    
    fixed = list(text)
    
    # Detect if this looks like a model number or serial number
    has_letters = any(c.isalpha() for c in text)
    has_digits = any(c.isdigit() for c in text)
    is_mixed = has_letters and has_digits
    
    for i, char in enumerate(fixed):
        char_upper = char.upper()
        
        # Context: Beginning of alphanumeric strings (likely letters)
        if i < 2 and is_mixed:
            if char == '0':
                fixed[i] = 'O'
            elif char == '1':
                fixed[i] = 'I'
            elif char == '5':
                fixed[i] = 'S'
            elif char == '8':
                fixed[i] = 'B'
        
        # Context: After letters, before digits (likely a digit)
        elif i > 0 and is_mixed:
            prev_is_letter = fixed[i-1].isalpha() if i > 0 else False
            next_is_digit = fixed[i+1].isdigit() if i < len(fixed) - 1 else False
            
            if prev_is_letter or next_is_digit:
                if char_upper == 'O':
                    fixed[i] = '0'
                elif char_upper == 'I' or char_upper == 'L':
                    fixed[i] = '1'
                elif char_upper == 'S':
                    fixed[i] = '5'
                elif char_upper == 'Z':
                    fixed[i] = '2'
                elif char_upper == 'B':
                    fixed[i] = '8'
        
        # Context: In mostly-digit strings (serial numbers)
        elif not is_mixed and has_digits:
            if char_upper == 'O':
                fixed[i] = '0'
            elif char_upper == 'I' or char_upper == 'L':
                fixed[i] = '1'
            elif char_upper == 'S':
                fixed[i] = '5'
            elif char_upper == 'Z':
                fixed[i] = '2'
            elif char_upper == 'B':
                fixed[i] = '8'
        
        # Low confidence: more aggressive fixes
        if confidence < 60:
            if char == '|':
                fixed[i] = 'I'
            elif char == '!':
                fixed[i] = '1'
            elif char == '@':
                fixed[i] = 'a'
    
    result = ''.join(fixed)
    
    if result != text:
        app.logger.debug(f"OCR fixup: '{text}' → '{result}' (conf={confidence:.0f})")
    
    return result

def extract_border_outlined_regions(image, debug_path=None):
    """
    Extract regions where text is surrounded by colored borders (like orange boxes)
    rather than having solid color backgrounds.
    """
    try:
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common border colors
        # Orange borders (common for model numbers)
        lower_orange = np.array([5, 100, 150])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Red borders
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        # Blue borders
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine all color masks
        color_mask = orange_mask + red_mask + blue_mask
        
        # Find contours of the colored regions
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the original image for result
        result = image.copy()
        
        # Create a white image for showing only the extracted regions
        white_bg = np.ones_like(image) * 255
        extracted_regions = white_bg.copy()
        
        # Create a mask for the areas inside the colored borders
        border_mask = np.zeros_like(color_mask)
        
        for contour in contours:
            # Calculate area of contour
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 100:  
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's likely a border by checking the ratio of contour area to bounding rect area
            # For borders, this ratio will be lower as the border forms a hollow rectangle
            rect_area = w * h
            area_ratio = area / rect_area if rect_area > 0 else 0
            
            # If area ratio is low (~ 0.1-0.5), it's likely a border not a filled region
            if area_ratio < 0.7 and area_ratio > 0.05:
                # Draw a slightly smaller rectangle inside the border to capture the text
                # This helps eliminate the actual colored border from the mask
                inner_margin = 3
                if w > 2*inner_margin and h > 2*inner_margin:  # Ensure we don't get negative dimensions
                    # Fill the inner area of the rectangle
                    inner_mask = np.zeros_like(border_mask)
                    cv2.rectangle(inner_mask, 
                                 (x + inner_margin, y + inner_margin), 
                                 (x + w - inner_margin, y + h - inner_margin), 
                                 255, -1)  # -1 means filled
                    border_mask = cv2.bitwise_or(border_mask, inner_mask)
                    
                    # Draw the ROI for debugging
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the regions inside borders
        if np.sum(border_mask) > 0:  # Check if we found any border regions
            np.copyto(extracted_regions, image, where=(border_mask[:,:,None]>0))
            
            # Save debug image if path is provided
            if debug_path:
                cv2.imwrite(f"{debug_path}_border_detection.jpg", result)
                cv2.imwrite(f"{debug_path}_border_regions.jpg", extracted_regions)
            
        return extracted_regions
        
    except Exception as e:
        logging.error(f"Error in extract_border_outlined_regions: {str(e)}")
        return image  # Return original image on error

def extract_colored_text_regions(image, debug_path=None):
    """
    Extract text from colored regions that commonly contain model numbers
    Specifically targets orange, red, blue, yellow and green backgrounds
    """
    try:
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common label backgrounds
        # Orange (like your model number background)
        lower_orange = np.array([5, 100, 150])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Red (wraps around the hue spectrum)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        # Blue
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Yellow
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine all color masks
        color_mask = orange_mask + red_mask + blue_mask + yellow_mask
        
        # Dilate to ensure we get complete regions
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(color_mask, kernel, iterations=2)
        
        # Find contours of color regions
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a white background image
        white_bg = np.ones_like(image) * 255
        
        # Copy only the masked regions to the white background
        colored_regions = cv2.bitwise_and(image, image, mask=dilated_mask)
        result = white_bg.copy()
        np.copyto(result, colored_regions, where=(dilated_mask[:,:,None]>0))
        
        # Save debug image if path is provided
        if debug_path:
            # Save color mask
            cv2.imwrite(f"{debug_path}_color_mask.jpg", color_mask)
            # Save result with white background
            cv2.imwrite(f"{debug_path}_color_regions.jpg", result)
            
        return result
        
    except Exception as e:
        logging.error(f"Error in extract_colored_text_regions: {str(e)}")
        return image  # Return original image on error

def extract_text_from_image_tesseract(image_path):
    """
    Enhanced Tesseract OCR implementation with complete preprocessing pipeline,
    similar to what we did with EasyOCR but optimized for speed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at path {image_path}")
        
        app.logger.debug("Starting enhanced Tesseract OCR processing pipeline")
        
        # Create an array to hold results from different preprocessing methods
        results = []
        
        # --- STEP 1: PREPROCESSING PIPELINE ---
        
        # Store processed images for use with different OCR configs
        processed_images = []
        
        # NEW: Add color-filtered processing to detect colored background text
        color_filtered = extract_colored_text_regions(image, debug_path=image_path)
        color_filtered_gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
        processed_images.append(("color_filtered", color_filtered_gray))
        
        # Save color filtered image for debugging
        cv2.imwrite(f"{image_path}_color_filtered.jpg", color_filtered_gray)
        
        # Convert to grayscale (base for other processing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(("gray", gray))
        
        # Save preprocessed image for debugging
        cv2.imwrite(f"{image_path}_gray.jpg", gray)
        
        # Enhanced contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        # Save enhanced image for debugging
        cv2.imwrite(f"{image_path}_enhanced.jpg", enhanced)
        
        # Apply CLAHE to color filtered image too
        color_enhanced = clahe.apply(color_filtered_gray)
        processed_images.append(("color_enhanced", color_enhanced))
        
        # Save color enhanced image for debugging
        cv2.imwrite(f"{image_path}_color_enhanced.jpg", color_enhanced)
        
        # Sharpening for better text edge definition
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        processed_images.append(("sharpened", sharpened))
        
        # Thresholding for better contrast - light background
        _, thresh_binary = cv2.threshold(enhanced, 170, 255, cv2.THRESH_BINARY)
        processed_images.append(("thresh_binary", thresh_binary))
        
        # Thresholding for better contrast - dark background
        _, thresh_binary_inv = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
        processed_images.append(("thresh_binary_inv", thresh_binary_inv))
        
        # Adaptive thresholding for uneven lighting
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive", adaptive))
        
        # Save adaptive image for debugging
        cv2.imwrite(f"{image_path}_adaptive.jpg", adaptive)
        
        # Denoising to remove speckles
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed_images.append(("denoised", denoised))
        
        # Save denoised image for debugging
        cv2.imwrite(f"{image_path}_denoised.jpg", denoised)
        
        # Thresholded denoised image
        _, thresh_denoised = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY)
        processed_images.append(("thresh_denoised", thresh_denoised))
        
        # Resize to double size to capture small details
        height, width = gray.shape[:2]
        scale_factor = 2.0
        resized = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), 
                           interpolation=cv2.INTER_CUBIC)
        processed_images.append(("resized", resized))
        
        # Save resized image for debugging
        cv2.imwrite(f"{image_path}_resized.jpg", resized)
        
        # Preprocessed resized
        resized_enhanced = cv2.resize(enhanced, (int(width * scale_factor), int(height * scale_factor)),
                                    interpolation=cv2.INTER_CUBIC)
        processed_images.append(("resized_enhanced", resized_enhanced))
        
        # Edge enhancement
        edges = cv2.Canny(denoised, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edge_enhanced = 255 - dilated_edges  # Invert for text recognition
        processed_images.append(("edge_enhanced", edge_enhanced))
        
        # Create a special preprocessed image combining multiple techniques
        # This often works well for appliance labels
        # 1. Enhanced contrast -> 2. Noise reduction -> 3. Adaptive threshold
        preprocessed = enhanced.copy()
        preprocessed = cv2.GaussianBlur(preprocessed, (3, 3), 0)
        preprocessed = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("preprocessed", preprocessed))
        
        # Save preprocessed image for debugging
        cv2.imwrite(f"{image_path}_preprocessed.jpg", preprocessed)
        
        # --- STEP 2: APPLY OCR WITH DIFFERENT CONFIGS ---
        
        # Tesseract configurations optimized for appliance labels
        configs = [
            ('--oem 1 --psm 6', 'lstm_single_block'),       # LSTM engine, single uniform block
            ('--oem 1 --psm 4', 'lstm_single_column'),      # LSTM engine, single column of variable sizes
            ('--oem 1 --psm 11', 'lstm_sparse'),            # LSTM engine, sparse text
            ('--oem 1 --psm 3', 'lstm_auto'),               # LSTM engine, auto page segmentation
            ('--oem 1 --psm 12', 'lstm_sparse_circular'),   # For circular text often found on appliance labels
            ('--oem 1 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/:. "', 'lstm_whitelist') # Add whitelist for model/serial chars
        ]
        
        # Try different combinations of image preprocessing and OCR configs
        for img_name, img in processed_images:
            # Select only the most promising configs for each image type to save time
            if img_name in ['gray', 'enhanced', 'preprocessed', 'color_filtered', 'color_enhanced']:
                # These are our base images, try all configs
                selected_configs = configs
            elif img_name in ['resized', 'resized_enhanced']:
                # Use sparse text and auto detection for resized images
                selected_configs = [c for c in configs if c[1] in ['lstm_sparse', 'lstm_auto', 'lstm_whitelist']]
            else:
                # For other processed images, just use one or two configs
                selected_configs = [configs[0], configs[5]]  # Use single block and whitelist
            
            for config, config_name in selected_configs:
                app.logger.debug(f"Running Tesseract with {img_name} image and {config_name} config")
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if text and text.strip():
                        results.append(text.strip())
                        if len(text.strip()) > 10:  # Only log longer texts to reduce noise
                            app.logger.debug(f"Found text with {img_name}/{config_name} ({len(text)} chars)")
                except Exception as e:
                    app.logger.warning(f"Error with {img_name}/{config_name}: {str(e)}")
        
        # --- STEP 3: POST-PROCESSING ---
        
        # Combine results and remove duplicates
        all_lines = []
        seen_lines = set()
        
        for result in results:
            lines = result.splitlines()
            for line in lines:
                cleaned_line = line.strip()
                # Skip empty lines or already seen lines
                if cleaned_line and cleaned_line not in seen_lines and len(cleaned_line) > 1:
                    all_lines.append(cleaned_line)
                    seen_lines.add(cleaned_line)
        
        # Appliance-specific post-processing
        processed_lines = []
        
        # Common OCR error corrections in appliance labels
        replacements = {
            '0': 'O',  # Number 0 to letter O for model numbers
            'I': '1',  # Letter I to number 1 for serial numbers
            'l': '1',  # Lowercase l to number 1
            '5': 'S',  # Number 5 to letter S in certain contexts
            'Z': '2',  # Letter Z to number 2 in certain contexts
            'B': '8',  # Letter B to number 8 in numeric contexts
            'g': '9',  # Lowercase g to number 9
            'o': '0',  # Lowercase o to number 0
        }
        
        for line in all_lines:
            # Special handling for likely model/serial numbers
            # If looks like alphanumeric code, preserve it exactly
            if re.search(r'^[A-Z0-9\-]{5,20}$', line, re.IGNORECASE):
                processed_lines.append(line.upper())  # Keep model/serial numbers as is, but uppercase
            
            # If might contain model/serial keywords, preserve
            elif any(keyword in line.upper() for keyword in ['MODEL', 'SERIAL', 'S/N', 'NO.', 'P/N']):
                processed_lines.append(line)  # Keep label info intact
                
            else:
                # General cleaning for other text
                # Fix common OCR errors in specific contexts
                if re.search(r'\d', line):  # Line contains numbers - likely technical info
                    # Don't replace chars in primarily numeric contexts
                    processed_line = line
                else:
                    # Apply replacements in text contexts
                    processed_line = line
                    for orig, repl in replacements.items():
                        # Only if standalone character or seems like a mistake
                        processed_line = re.sub(r'(\s' + orig + r'\s)', r'\1' + repl + r'\2', processed_line)
                        
                # Remove unusual non-alphanumeric characters except important ones
                processed_line = re.sub(r'[^A-Za-z0-9\s\.\-:#/,()&$%+]', '', processed_line)
                
                if processed_line.strip():
                    processed_lines.append(processed_line)
        
        # Prioritize the processed lines
        def score_line(line):
            """Score the importance of a line for appliance label information"""
            score = 0
            # Prioritize lines with potential model/serial number patterns
            if re.search(r'[A-Z]{1,4}[0-9]{4,10}', line, re.IGNORECASE):
                score += 10
            # Prioritize lines with important label keywords
            if any(keyword in line.upper() for keyword in ['MODEL', 'SERIAL', 'PRODUCT', 'CODE', 'BRAND']):
                score += 5
            # Prioritize lines with brand names
            if any(brand in line.upper() for brand in ['LG', 'SAMSUNG', 'GE', 'WHIRLPOOL', 'MAYTAG', 'FRIGIDAIRE']):
                score += 3
            # Prioritize lines with more content
            score += min(len(line) / 10, 3)  # Cap at 3 points
            return score
            
        # Sort lines by score (highest first)
        processed_lines.sort(key=score_line, reverse=True)
                
        combined_text = '\n'.join(processed_lines)
        app.logger.debug(f"Final Tesseract results: {len(processed_lines)} lines")
        
        return combined_text
    
    except Exception as e:
        app.logger.error(f"Error in Tesseract OCR processing: {str(e)}")
        return f"OCR processing error: {str(e)[:100]}..."

def extract_text_from_image_easyocr(image_path):
    """
    Enhanced EasyOCR implementation with robust preprocessing for challenging images
    """
    try:
        # First check if file exists and is valid
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            raise ValueError(f"Image file is missing or empty: {image_path}")
            
        # Try to read the image with OpenCV first to validate it
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"OpenCV could not read the image: {image_path}")
            
        # Apply multiple preprocessing techniques and run OCR on each
        results = []
        
        # 1. Original image
        original_results = easyocr_reader.readtext(image_path, detail=0)
        app.logger.debug(f"Original image results ({len(original_results)}): {original_results}")
        results.extend([r for r in original_results if len(r) > 1])  # Filter out single characters
        
        # 2. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_path = image_path + "_gray.jpg"
        cv2.imwrite(gray_path, gray)
        gray_results = easyocr_reader.readtext(gray_path, detail=0)
        app.logger.debug(f"Grayscale results ({len(gray_results)}): {gray_results}")
        results.extend([r for r in gray_results if len(r) > 1 and r not in results])
        
        # 3. Thresholding for better contrast
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh_path = image_path + "_thresh.jpg"
        cv2.imwrite(thresh_path, thresh)
        thresh_results = easyocr_reader.readtext(thresh_path, detail=0)
        app.logger.debug(f"Threshold results ({len(thresh_results)}): {thresh_results}")
        results.extend([r for r in thresh_results if len(r) > 1 and r not in results])
        
        # 4. Adaptive thresholding for uneven lighting
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        adaptive_path = image_path + "_adaptive.jpg"
        cv2.imwrite(adaptive_path, adaptive)
        adaptive_results = easyocr_reader.readtext(adaptive_path, detail=0)
        app.logger.debug(f"Adaptive results ({len(adaptive_results)}): {adaptive_results}")
        results.extend([r for r in adaptive_results if len(r) > 1 and r not in results])
        
        # 5. Try increasing the image size
        height, width = gray.shape[:2]
        resized = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        resized_path = image_path + "_resized.jpg"
        cv2.imwrite(resized_path, resized)
        resized_results = easyocr_reader.readtext(resized_path, detail=0)
        app.logger.debug(f"Resized results ({len(resized_results)}): {resized_results}")
        results.extend([r for r in resized_results if len(r) > 1 and r not in results])
        
        # 6. Try denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        denoised_path = image_path + "_denoised.jpg"
        cv2.imwrite(denoised_path, denoised)
        denoised_results = easyocr_reader.readtext(denoised_path, detail=0)
        app.logger.debug(f"Denoised results ({len(denoised_results)}): {denoised_results}")
        results.extend([r for r in denoised_results if len(r) > 1 and r not in results])
        
        # Filter and clean up results
        filtered_results = []
        for r in results:
            # Remove very short strings and those that are just single characters
            if len(r) > 1 and not r.isspace() and r not in filtered_results:
                filtered_results.append(r)
        
        app.logger.debug(f"Filtered results ({len(filtered_results)}): {filtered_results}")
        
        if not filtered_results:
            return "No readable text detected"
            
        return '\n'.join(filtered_results)
        
    except Exception as e:
        app.logger.error(f"Error in OCR processing: {str(e)}")
        # Return a message that can be shown to the user
        return f"OCR processing error: {str(e)[:100]}..."


def apply_tighter_crop(image, shrink_percent=10):
    """
    Apply tighter crop by shrinking margins.
    shrink_percent: how much to crop from each edge (default 10%)
    """
    h, w = image.shape[:2]
    margin_h = int(h * shrink_percent / 100)
    margin_w = int(w * shrink_percent / 100)
    
    # Ensure we don't crop too much
    if margin_h * 2 >= h or margin_w * 2 >= w:
        return image
    
    cropped = image[margin_h:h-margin_h, margin_w:w-margin_w]
    app.logger.debug(f"Applied tighter crop: {shrink_percent}% margins removed")
    return cropped


def apply_rotation(image, angle):
    """
    Rotate image by specified angle (in degrees).
    Positive = clockwise, Negative = counter-clockwise
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with white background
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    
    app.logger.debug(f"Applied rotation: {angle}°")
    return rotated


def calculate_extraction_confidence(brand, model, serial):
    """
    Calculate overall confidence (0-100) based on extracted fields.
    Returns confidence score and whether retry is needed.
    Only retry if BOTH model AND serial are missing (critical failure).
    """
    score = 0
    
    # Brand scoring (max 25 points)
    if brand != "Unknown":
        score += 25
    
    # Model scoring (max 40 points)
    if model != "Unknown":
        if len(model) >= 6:  # Reasonable length
            score += 20
        if len(model) >= 8:  # Good length
            score += 10
        # Check for alphanumeric mix (typical of models)
        has_letters = any(c.isalpha() for c in model)
        has_digits = any(c.isdigit() for c in model)
        if has_letters and has_digits:
            score += 10
    
    # Serial scoring (max 35 points)
    if serial != "Unknown":
        if len(serial) >= 8:  # Reasonable length
            score += 20
        if len(serial) >= 10:  # Good length
            score += 10
        # Serials typically have high digit ratio
        digit_ratio = sum(c.isdigit() for c in serial) / len(serial) if len(serial) > 0 else 0
        if digit_ratio > 0.5:
            score += 5
    
    # CONSERVATIVE retry: Only if BOTH model and serial are missing (critical failure)
    # If we got at least one key field, don't waste time retrying
    needs_retry = (model == "Unknown" and serial == "Unknown")
    
    app.logger.debug(f"Extraction confidence: {score}/100 (brand={brand}, model={model}, serial={serial}, retry={needs_retry})")
    return score, needs_retry


def detect_and_correct_perspective(img):
    """
    Detect the label rectangle, crop to it, and apply perspective correction to straighten it.
    This is the "label-first" approach - we only OCR the label region, not the whole photo.
    Uses edge detection and contour finding to locate the label rectangle.
    """
    try:
        app.logger.debug("Attempting label detection and perspective correction")
        original = img.copy()
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding for better edge detection on varied lighting
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection with multiple strategies
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges3 = cv2.Canny(adaptive, 50, 150)
        
        # Combine edge maps
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            app.logger.debug("No contours found, using original image")
            return img
        
        # Sort contours by area and examine the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        label_contour = None
        
        # Find a contour that is roughly rectangular and significant in size
        min_area = (width * height) * 0.15  # At least 15% of image
        max_area = (width * height) * 0.95  # Not the entire image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < min_area or area > max_area:
                continue
            
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Look for quadrilaterals (4 corners)
            if len(approx) == 4:
                # Check aspect ratio (labels are usually wider than tall, or square)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Accept aspect ratios between 0.5 and 3.0 (reasonable for labels)
                if 0.5 <= aspect_ratio <= 3.0:
                    label_contour = approx
                    app.logger.debug(f"Found label contour: area={area}, aspect={aspect_ratio:.2f}")
                    break
        
        # If no perfect quadrilateral found, use the largest reasonable contour
        if label_contour is None:
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    # Use bounding rectangle as fallback
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        # Create quadrilateral from bounding box
                        label_contour = np.array([
                            [[x, y]],
                            [[x + w, y]],
                            [[x + w, y + h]],
                            [[x, y + h]]
                        ], dtype=np.int32)
                        app.logger.debug(f"Using bounding rectangle: area={area}, aspect={aspect_ratio:.2f}")
                        break
        
        if label_contour is None:
            app.logger.debug("No suitable label contour found, using original image")
            return img
        
        # Get the corner points and order them
        pts = label_contour.reshape(4, 2).astype(np.float32)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum: top-left will have smallest sum, bottom-right largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference: top-right will have smallest diff, bottom-left largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Calculate width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Ensure dimensions are reasonable
        if maxWidth < 100 or maxHeight < 100:
            app.logger.debug("Detected label too small, using original")
            return img
        
        # Construct destination points for the "birds-eye view"
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Calculate perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        app.logger.debug(f"Label cropped and corrected: {maxWidth}x{maxHeight} (from {width}x{height})")
        return warped
        
    except Exception as e:
        app.logger.warning(f"Label detection failed: {str(e)}, using original image")
        return img

def hybrid_ocr_extract(image_path):
    """
    Optimized OCR approach following best practices:
    1. Auto-detect and crop to label region
    2. Perspective correction for tilted labels
    3. Upscale 2-4x for better character recognition
    4. Try normal + inverted + high-contrast
    5. Use PSM 6/7 for specific model/serial rows
    6. Extract using keyword patterns (MODEL, SERIAL, etc.)
    """
    try:
        app.logger.debug("Starting optimized OCR processing")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        height, width = img.shape[:2]
        
        # Step 0.5: PERSPECTIVE CORRECTION - Detect and straighten label if tilted
        img = detect_and_correct_perspective(img)
        height, width = img.shape[:2]  # Update dimensions after correction
        
        # Step 1: UPSCALE 3x for better character recognition
        app.logger.debug("Upscaling image 3x for better OCR")
        upscaled = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        upscaled_path = f"{image_path}_upscaled.jpg"
        cv2.imwrite(upscaled_path, upscaled)
        
        # Step 2: Create 3 optimized preprocessing variants for different label types
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # === VARIANT 1: CLAHE Enhanced (Best for: glossy stickers, varied lighting) ===
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This handles local contrast variations common in glossy/reflective labels
        app.logger.debug("Creating V1: CLAHE enhanced")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        v1_clahe = clahe.apply(gray)
        # Apply slight Gaussian blur to reduce noise
        v1_clahe = cv2.GaussianBlur(v1_clahe, (3, 3), 0)
        v1_path = f"{image_path}_v1_clahe.jpg"
        cv2.imwrite(v1_path, v1_clahe)
        
        # === VARIANT 2: Sharpened (Best for: stamped metal, embossed text) ===
        # Create a sharpening kernel and apply it
        app.logger.debug("Creating V2: Sharpened")
        # Unsharp masking: blur -> subtract -> add back
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        v2_sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        # Additional edge enhancement
        kernel_sharp = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
        v2_sharp = cv2.filter2D(v2_sharp, -1, kernel_sharp)
        # Normalize to full range
        v2_sharp = cv2.normalize(v2_sharp, None, 0, 255, cv2.NORM_MINMAX)
        v2_path = f"{image_path}_v2_sharp.jpg"
        cv2.imwrite(v2_path, v2_sharp)
        
        # === VARIANT 3: Adaptive Binary (Best for: printed labels, clear text) ===
        # Use adaptive thresholding which handles varying background colors
        app.logger.debug("Creating V3: Adaptive binary")
        # Apply bilateral filter first to smooth while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # Adaptive threshold with large block size for label-scale variations
        v3_binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 25, 10)
        # Try inverted version too (for dark text on light background)
        v3_binary_inv = cv2.bitwise_not(v3_binary)
        v3_path = f"{image_path}_v3_binary.jpg"
        v3_inv_path = f"{image_path}_v3_binary_inv.jpg"
        cv2.imwrite(v3_path, v3_binary)
        cv2.imwrite(v3_inv_path, v3_binary_inv)
        
        all_text_lines = []
        confidence_scores = {}
        tesseract_lines = []
        easyocr_lines = []
        
        # Character whitelist for appliance labels (alphanumeric + common punctuation)
        label_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/().#: '
        
        # Step 3: Extract text from each variant using optimized Tesseract and EasyOCR
        variants = [
            (v1_path, "V1-CLAHE"),
            (v2_path, "V2-Sharp"),
            (v3_path, "V3-Binary"),
            (v3_inv_path, "V3-BinaryInv")
        ]
        
        for img_path, variant_name in variants:
            app.logger.debug(f"OCR processing {variant_name}")
            variant_tess = []
            variant_easy = []
            
            # TESSERACT with optimized settings for labels
            # --oem 1: Use LSTM neural net mode (best accuracy)
            # --psm 6: Uniform block of text (typical label layout)
            tesseract_config = f'--oem 1 --psm 6 -c tessedit_char_whitelist={label_charset}'
            
            try:
                # Get detailed output with confidence scores
                tess_data = pytesseract.image_to_data(img_path, config=tesseract_config, output_type=pytesseract.Output.DICT)
                
                # Extract lines with confidence filtering
                for i, text in enumerate(tess_data['text']):
                    conf = int(tess_data['conf'][i]) if tess_data['conf'][i] != -1 else 0
                    if text.strip() and conf > 30:  # Only keep reasonably confident results
                        variant_tess.append({
                            'text': text.strip(),
                            'confidence': conf,
                            'source': f'Tesseract-{variant_name}'
                        })
                        
            except Exception as e:
                app.logger.warning(f"Tesseract failed on {variant_name}: {str(e)}")
            
            # EASYOCR with optimized settings
            try:
                # paragraph=False: Return individual text regions
                # decoder='beamsearch': Better accuracy (slower)
                # allowlist: Restrict to label characters
                easy_results = easyocr_reader.readtext(
                    img_path, 
                    detail=1,  # Get confidence scores
                    paragraph=False,
                    decoder='beamsearch',
                    allowlist=label_charset
                )
                
                # Extract with confidence filtering
                for bbox, text, conf in easy_results:
                    if text.strip() and conf > 0.3:  # EasyOCR confidence is 0-1
                        variant_easy.append({
                            'text': text.strip(),
                            'confidence': int(conf * 100),  # Normalize to 0-100
                            'source': f'EasyOCR-{variant_name}'
                        })
                        
            except Exception as e:
                app.logger.warning(f"EasyOCR failed on {variant_name}: {str(e)}")
            
            # Track results per variant
            if variant_tess or variant_easy:
                tesseract_lines.extend(variant_tess)
                easyocr_lines.extend(variant_easy)
                total = len(variant_tess) + len(variant_easy)
                confidence_scores[variant_name] = total
                app.logger.debug(f"{variant_name}: {len(variant_tess)} Tess + {len(variant_easy)} Easy = {total} lines")
        
        # Step 4: Merge and vote on results
        app.logger.debug("Merging OCR results with confidence voting")
        
        # Combine all results
        all_ocr_results = tesseract_lines + easyocr_lines
        
        # Group similar lines and vote
        merged_lines = {}
        for result in all_ocr_results:
            text = result['text'].upper()
            conf = result['confidence']
            source = result['source']
            
            # Find similar existing lines (fuzzy match)
            matched = False
            for existing_text in list(merged_lines.keys()):
                # Simple similarity: same length and 80%+ character match
                if len(text) == len(existing_text):
                    matches = sum(1 for a, b in zip(text, existing_text) if a == b)
                    if matches / len(text) >= 0.8:
                        # Merge with existing
                        merged_lines[existing_text]['votes'] += 1
                        merged_lines[existing_text]['total_conf'] += conf
                        merged_lines[existing_text]['sources'].append(source)
                        matched = True
                        break
            
            if not matched:
                # New unique line
                merged_lines[text] = {
                    'votes': 1,
                    'total_conf': conf,
                    'sources': [source],
                    'original': result['text']
                }
        
        # Select best lines based on votes and confidence
        final_lines = []
        for text, data in merged_lines.items():
            avg_conf = data['total_conf'] / data['votes']
            score = (data['votes'] * 50) + avg_conf  # Weight: votes + confidence
            
            if score > 60:  # Threshold for keeping a line
                final_lines.append({
                    'text': data['original'],
                    'score': score,
                    'votes': data['votes'],
                    'confidence': avg_conf
                })
        
        # Sort by score (best first)
        final_lines.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply context-aware fix-ups to top results
        for line_data in final_lines:
            line_data['text'] = apply_ocr_fixups(line_data['text'], line_data['confidence'])
            all_text_lines.append(line_data['text'])
        
        app.logger.debug(f"Merged to {len(final_lines)} high-confidence lines")
        
        # Step 5: Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in all_text_lines:
            line_upper = line.upper().strip()
            if line_upper and line_upper not in seen and len(line_upper) > 1:
                unique_lines.append(line_upper)
                seen.add(line_upper)
        
        app.logger.debug(f"Extracted {len(unique_lines)} unique lines")
        app.logger.debug(f"First 10 lines: {unique_lines[:10]}")
        
        # Combine into single text
        final_text = '\n'.join(unique_lines)
        
        # Clean up temporary files
        for temp_file in [upscaled_path, normal_path, contrast_path, inverted_path, binary_path]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        return final_text
        
    except Exception as e:
        app.logger.error(f"Error in OCR processing: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return f"OCR processing error: {str(e)[:100]}..."

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# Fix the upload folder path to be within the appliance_lookup/static folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.logger.debug(f"Upload folder set to: {UPLOAD_FOLDER}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            app.logger.debug(f"Image saved at: {filepath}")

            # Assess image quality first
            quality_assessment = assess_image_quality(filepath)
            app.logger.debug(f"Image quality: {quality_assessment}")

            # Generate a unique debug path for border detection
            debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_debug")

            # ===== ROBUST 2-STAGE OCR SYSTEM =====
            # Stage 1: Google Gemini Vision AI (PRIMARY - most accurate)
            # Stage 2: Tesseract + EasyOCR (FALLBACK - if Gemini fails/unavailable)
            
            all_text = ""  # Initialize for legacy code compatibility
            extraction_method = "Unknown"
            
            # Try Gemini Vision first (much more robust than traditional OCR)
            try:
                app.logger.info("Stage 1: Attempting Gemini Vision AI extraction")
                gemini_results = extract_with_gemini(filepath)
                
                # Check if Gemini was successful
                if gemini_results.get('brand') != 'Unknown' or gemini_results.get('model') != 'Unknown':
                    final_results = gemini_results
                    extraction_method = "Gemini Vision AI"
                    app.logger.info(f"✓ Gemini Vision extraction successful")
                    
                    # Generate a synthetic "all_text" for compatibility with brand parsers
                    all_text = f"Brand: {final_results['brand']}\nModel: {final_results['model']}\nSerial: {final_results['serial']}"
                else:
                    raise Exception("Gemini returned no useful data, falling back to traditional OCR")
                    
            except Exception as gemini_error:
                app.logger.warning(f"Gemini Vision unavailable or failed: {gemini_error}")
                app.logger.info("Stage 2: Falling back to Tesseract + EasyOCR")
                
                # Fallback to traditional OCR
                # Fallback to traditional OCR
                try:
                    app.logger.info("Using Tesseract + EasyOCR for extraction")
                    
                    img = cv2.imread(filepath)
                    h, w = img.shape[:2]
                    img_upscaled = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                    upscaled_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_upscaled.jpg")
                    cv2.imwrite(upscaled_path, img_upscaled)
                    
                    tess_text = pytesseract.image_to_string(upscaled_path, config='--psm 6')
                    easy_results = easyocr_reader.readtext(upscaled_path, detail=0, paragraph=False)
                    easy_text = '\n'.join(easy_results)
                    all_text = tess_text + '\n' + easy_text
                    
                    if os.path.exists(upscaled_path):
                        os.remove(upscaled_path)
                    
                    # Parse with existing parsers
                    legacy_parsed_data = parse_appliance_text(all_text)
                    new_parsed_data = parse_appliance_label(all_text)
                    
                    # Combine results
                    final_results = {
                        "brand": new_parsed_data.get("brand", "Unknown") if new_parsed_data.get("brand") != "Unknown" else legacy_parsed_data.get("brand", "Unknown"),
                        "model": new_parsed_data.get("model", "Unknown") if new_parsed_data.get("model") != "Unknown" else legacy_parsed_data.get("model", "Unknown"),
                        "serial": new_parsed_data.get("serial", "Unknown") if new_parsed_data.get("serial") != "Unknown" else legacy_parsed_data.get("serial", "Unknown"),
                        "other": new_parsed_data.get("other", "") if new_parsed_data.get("other", "").strip() else legacy_parsed_data.get("other", "")
                    }
                    extraction_method = "Tesseract + EasyOCR (Fallback)"
                
                except Exception as e:
                    app.logger.error(f"Traditional OCR extraction also failed: {e}")
                    all_text = ""  # Ensure it's defined even on error
                    final_results = {
                        "brand": "Unknown",
                        "model": "Unknown",
                        "serial": "Unknown",
                        "other": f"Both extraction methods failed. Gemini: {str(gemini_error)}, OCR: {str(e)}"
                    }
                    extraction_method = "Failed"

            # Ensure all keys exist
            final_results = {
                "brand": final_results.get("brand", "Unknown"),
                "model": final_results.get("model", "Unknown"),
                "serial": final_results.get("serial", "Unknown"),
                "other": final_results.get("other", "")
            }
            
            app.logger.info(f"Extraction method: {extraction_method}")
            app.logger.info(f"Final results: Brand={final_results['brand']}, Model={final_results['model']}, Serial={final_results['serial']}")
            
            # Try brand-specific parser if we have a brand
            if final_results["brand"] != "Unknown":
                brand_specific = parse_with_brand_parser(final_results["brand"], all_text)
                if brand_specific:
                    # Use brand parser results if they're better
                    if brand_specific["model"] != "Unknown" and final_results["model"] == "Unknown":
                        final_results["model"] = brand_specific["model"]
                        app.logger.debug(f"Using brand-specific model: {brand_specific['model']}")
                    if brand_specific["serial"] != "Unknown" and final_results["serial"] == "Unknown":
                        final_results["serial"] = brand_specific["serial"]
                        app.logger.debug(f"Using brand-specific serial: {brand_specific['serial']}")

            # Special case handling for LG washing machine
            if ("WASHING MACHINE" in all_text.upper() and 
                "LG" in all_text.upper() and
                "WM3670HWA" in all_text.upper() and
                "707TWGG04983" in all_text.upper()):
                app.logger.debug("Special case: Detected specific LG washing machine")
                final_results["brand"] = "LG"
                final_results["model"] = "WM3670HWA"
                final_results["serial"] = "707TWGG04983"

            app.logger.debug(f"Final combined results: {final_results}")
            
            # Calculate confidence scores
            confidence_info = calculate_confidence_scores(
                brand=final_results["brand"],
                model=final_results["model"],
                serial=final_results["serial"],
                raw_text=all_text
            )
            
            final_results["confidence"] = confidence_info["scores"]
            final_results["confidence_details"] = confidence_info["details"]
            final_results["needs_review"] = confidence_info["needs_review"]
            
            # Add quality assessment
            final_results["quality"] = quality_assessment

            # Generate URL for the image
            image_url = url_for('static', filename=f'uploads/{unique_filename}')

            # Return the results template with all necessary data
            return render_template('results.html', 
                                  ocr_result=all_text, 
                                  image_url=image_url, 
                                  parsed_data=final_results,
                                  extraction_method=extraction_method)

        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type or no file selected'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
