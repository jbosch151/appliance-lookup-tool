import cv2
import pytesseract
import numpy as np
import easyocr
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the correct path for Tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text(image_path, use_easyocr=False, save_debug=False):
    """Extract text from an image using either EasyOCR or Tesseract"""
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image: {image_path}")
        return "Error: Image could not be read"
    
    if use_easyocr:
        # EasyOCR
        result = reader.readtext(image_path)
        return " ".join([res[1] for res in result])  # Combine detected text
    else:
        # Tesseract
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(gray, config=custom_config).strip()

def detect_text_in_colored_borders(image_path, save_debug=False):
    """
    Detect text surrounded by colored borders in the image.
    
    Args:
        image_path: Path to the input image
        save_debug: Whether to save debug images
        
    Returns:
        dict: Contains detected text grouped by border colors
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image: {image_path}")
        return {"error": "Image could not be read"}
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Convert to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for common border colors
    color_ranges = {
        'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],  # Red has two ranges
        'blue': [([100, 100, 100], [140, 255, 255])],
        'green': [([40, 40, 100], [80, 255, 255])],
        'yellow': [([20, 100, 100], [35, 255, 255])],
        'orange': [([10, 100, 100], [25, 255, 255])]
    }
    
    result = {}
    
    # Process each color
    for color_name, ranges in color_ranges.items():
        # Create mask for this color
        color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for lower, upper in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Get mask for this color range
            range_mask = cv2.inRange(hsv_image, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, range_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential borders
        border_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's a border-like shape (has some thickness but not too thick)
                if w > 20 and h > 20 and min(w, h) / max(w, h) > 0.1:
                    # Draw contour on visualization image
                    cv2.drawContours(vis_image, [contour], -1, (0, 255, 255), 2)
                    
                    # Create a slightly smaller ROI inside the contour to capture text
                    padding = int(min(w, h) * 0.1)
                    inner_x = x + padding
                    inner_y = y + padding
                    inner_w = w - 2*padding
                    inner_h = h - 2*padding
                    
                    # Make sure inner dimensions are valid
                    if inner_w > 0 and inner_h > 0:
                        # Extract ROI from the original image
                        roi = image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
                        
                        # Convert ROI to grayscale for OCR
                        if roi.size > 0:
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            
                            # Draw ROI rectangle on visualization image
                            cv2.rectangle(vis_image, (inner_x, inner_y), 
                                         (inner_x+inner_w, inner_y+inner_h), (0, 0, 255), 1)
                            
                            # Extract text from ROI
                            custom_config = r'--oem 3 --psm 6'
                            text = pytesseract.image_to_string(roi_gray, config=custom_config).strip()
                            
                            if text:
                                if color_name not in result:
                                    result[color_name] = []
                                result[color_name].append({
                                    'text': text,
                                    'bbox': [inner_x, inner_y, inner_x+inner_w, inner_y+inner_h],
                                    'confidence': 1.0  # Placeholder since Tesseract doesn't return confidence
                                })
                                
                                # Add text annotation to visualization
                                cv2.putText(vis_image, text[:10] + ('...' if len(text) > 10 else ''),
                                          (inner_x, inner_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    # Save debug image if requested
    if save_debug:
        debug_path = f"{image_path}_border_detection_debug.jpg"
        cv2.imwrite(debug_path, vis_image)
        logging.info(f"Saved debug image to {debug_path}")
    
    return result

def enhanced_text_extraction(image_path, save_debug=False):
    """
    Perform enhanced text extraction using both regular OCR and border color detection
    
    Args:
        image_path: Path to the input image
        save_debug: Whether to save debug images
    
    Returns:
        dict: Contains all extracted text and metadata
    """
    # Get basic OCR text
    basic_text = extract_text(image_path)
    
    # Get text in colored borders
    border_text = detect_text_in_colored_borders(image_path, save_debug)
    
    # Combine results
    result = {
        "full_text": basic_text,
        "border_text": border_text
    }
    
    # For convenience, flatten all text from colored borders into a single string as well
    all_border_text = []
    for color, entries in border_text.items():
        for entry in entries:
            all_border_text.append(entry['text'])
    
    result["all_border_text"] = " ".join(all_border_text)
    
    return result

# Example usage when file is run directly
if __name__ == "__main__":
    image_path = 'model_ocr/sample_plate.jpg'
    results = enhanced_text_extraction(image_path, save_debug=True)
    print("\nFull OCR Text:")
    print(results["full_text"])
    print("\nText in Colored Borders:")
    for color, entries in results["border_text"].items():
        print(f"\n{color.upper()} BORDER:")
        for entry in entries:
            print(f"- {entry['text']}")
