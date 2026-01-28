"""
PaddleOCR-based extraction for fast local processing.
This is Stage 1 of the 2-stage pipeline.
"""
import logging
import cv2
import numpy as np
from typing import Dict, Tuple

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not installed. Local OCR will not work.")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PaddleOCR once (lazy loading)
_paddle_reader = None


def get_paddle_reader():
    """Get or initialize the PaddleOCR reader."""
    global _paddle_reader
    if _paddle_reader is None and PADDLE_AVAILABLE:
        _paddle_reader = PaddleOCR(
            use_angle_cls=True,  # Enable angle classification for rotated text
            lang='en'
        )
        logging.info("PaddleOCR initialized")
    return _paddle_reader


def extract_with_paddle(image_path: str) -> Tuple[Dict[str, str], int]:
    """
    Extract text from image using PaddleOCR and parse for appliance fields.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (results_dict, confidence_score)
        - results_dict: {'brand', 'model', 'serial', 'other'}
        - confidence_score: 0-100 indicating extraction quality
    """
    if not PADDLE_AVAILABLE:
        logging.error("PaddleOCR not available")
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': 'PaddleOCR not installed'
        }, 0
    
    try:
        reader = get_paddle_reader()
        
        # Read and preprocess image (3x upscale for better OCR)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        img_upscaled = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        logging.debug(f"PaddleOCR processing image: {image_path}")
        
        # Run PaddleOCR
        result = reader.ocr(img_upscaled)
        
        # Extract text and confidence scores
        all_text = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # Extracted text
                    conf = line[1][1]  # Confidence score
                    all_text.append(text)
                    confidences.append(conf)
        
        combined_text = '\n'.join(all_text)
        avg_confidence = int(np.mean(confidences) * 100) if confidences else 0
        
        logging.debug(f"PaddleOCR extracted {len(all_text)} lines, avg confidence: {avg_confidence}%")
        logging.debug(f"First 300 chars: {combined_text[:300]}")
        
        # Parse the extracted text using existing parsers
        from parser_module import parse_appliance_info
        from appliance_label_parser import parse_label
        
        # Try both parsers
        parser1_result = parse_appliance_info(combined_text)
        parser2_result = parse_label(combined_text)
        
        # Combine results (prefer more complete parser)
        def count_found(r):
            return sum(1 for v in [r.get('brand'), r.get('model'), r.get('serial')] 
                      if v and v != 'Unknown')
        
        if count_found(parser2_result) >= count_found(parser1_result):
            final_result = parser2_result
        else:
            final_result = parser1_result
        
        # Calculate field-level confidence
        field_confidence = calculate_extraction_confidence(final_result, avg_confidence)
        
        logging.info(f"PaddleOCR extraction: Brand={final_result.get('brand')}, "
                    f"Model={final_result.get('model')}, Serial={final_result.get('serial')}, "
                    f"Confidence={field_confidence}%")
        
        return final_result, field_confidence
        
    except Exception as e:
        logging.error(f"PaddleOCR extraction failed: {e}")
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': f'PaddleOCR error: {str(e)}'
        }, 0


def calculate_extraction_confidence(result: Dict[str, str], ocr_confidence: int) -> int:
    """
    Calculate overall confidence based on OCR confidence and field quality.
    
    Args:
        result: Dictionary with extracted fields
        ocr_confidence: Average OCR confidence (0-100)
    
    Returns:
        Overall confidence score (0-100)
    """
    score = ocr_confidence * 0.4  # OCR confidence contributes 40%
    
    brand = result.get('brand', '')
    model = result.get('model', '')
    serial = result.get('serial', '')
    
    # Brand quality (20% weight)
    if brand and brand != 'Unknown':
        known_brands = ['GE', 'Whirlpool', 'Samsung', 'Maytag', 'Frigidaire', 
                       'LG', 'Bosch', 'KitchenAid', 'Electrolux', 'Kenmore']
        if any(kb.lower() in brand.lower() for kb in known_brands):
            score += 20
        elif len(brand) >= 3:
            score += 10
    
    # Model quality (25% weight)
    if model and model != 'Unknown' and len(model) >= 4:
        model_clean = model.replace(' ', '').replace('-', '')
        # Good models have 6-20 chars with letters and numbers
        if 6 <= len(model_clean) <= 20:
            digits = sum(c.isdigit() for c in model_clean)
            letters = sum(c.isalpha() for c in model_clean)
            if digits > 0 and letters > 0:
                score += 25
            else:
                score += 15
        elif 4 <= len(model_clean) <= 25:
            score += 15
        else:
            score += 5
    
    # Serial quality (15% weight)
    if serial and serial != 'Unknown' and len(serial) >= 4:
        serial_clean = serial.replace(' ', '').replace('-', '')
        if 6 <= len(serial_clean) <= 20 and serial_clean.isalnum():
            score += 15
        elif 4 <= len(serial_clean) <= 25:
            score += 10
        else:
            score += 5
    
    return min(100, max(0, int(score)))


if __name__ == '__main__':
    # Test with a sample image
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result, confidence = extract_with_paddle(test_image)
        print(f"\nPaddleOCR Extraction Results:")
        print(f"Brand: {result['brand']}")
        print(f"Model: {result['model']}")
        print(f"Serial: {result['serial']}")
        print(f"Confidence: {confidence}%")
        if confidence < 75:
            print("\n⚠️  Low confidence - would trigger Gemini fallback")
    else:
        print("Usage: python paddle_ocr.py <image_path>")
