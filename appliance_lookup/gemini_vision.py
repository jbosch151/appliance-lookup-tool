"""
Google Gemini Vision API integration for appliance label extraction.
This is the fallback for when local OCR confidence is low.
"""
import logging
import os
import base64
from typing import Dict, Optional
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Gemini fallback will not work.")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def configure_gemini(api_key: Optional[str] = None):
    """Configure Gemini API with the provided key or from environment."""
    if not GEMINI_AVAILABLE:
        return False  # Return False instead of raising
    
    key = api_key or os.getenv('GOOGLE_API_KEY')
    if not key:
        logging.warning("No Gemini API key found - fallback will not be available")
        return False  # Return False instead of raising
    
    # DEBUG: Log last 4 chars to verify which key Railway is using
    logging.info(f"API key detected ending in: ...{key[-4:]}")
    
    genai.configure(api_key=key)
    logging.info("Gemini API configured successfully")
    return True


def extract_with_gemini(image_path: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Extract brand, model, and serial from appliance label using Google Gemini Vision.
    
    Args:
        image_path: Path to the image file
        api_key: Optional Google API key (if not set in environment)
    
    Returns:
        Dictionary with 'brand', 'model', 'serial', and 'other' keys
    """
    if not GEMINI_AVAILABLE:
        logging.error("Gemini not available, returning unknown values")
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': 'Gemini API not configured'
        }
    
    try:
        # Configure API if not already done
        if not configure_gemini(api_key):
            # No API key available
            logging.warning("Gemini API not configured - no API key available")
            return {
                'brand': 'Unknown',
                'model': 'Unknown',
                'serial': 'Unknown',
                'other': 'Gemini API key not configured'
            }
        
        # Use Gemini Flash (latest) - better quota limits
        model = genai.GenerativeModel('gemini-flash-latest')
        logging.info("Using Gemini Flash Latest model with new API key")
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create prompt for structured extraction
        prompt = """Extract the following information from this appliance label image:

1. BRAND: The manufacturer name (GE, Whirlpool, Samsung, Maytag, Frigidaire, etc.)
2. MODEL: The model number (usually after "MODEL", "MODEL NO", "MODEL #", or similar)
3. SERIAL: The serial number (usually after "SERIAL", "SERIAL NO", "SER", "S/N", or similar)

Return ONLY a JSON object in this exact format:
{
    "brand": "extracted brand or Unknown",
    "model": "extracted model number or Unknown",
    "serial": "extracted serial number or Unknown"
}

Important:
- Extract the EXACT text as it appears on the label
- Do not include keywords like "MODEL" or "SERIAL" in the extracted values
- If you cannot find a field, use "Unknown"
- Return ONLY valid JSON, no other text
"""
        
        logging.debug(f"Sending image to Gemini Vision API: {image_path}")
        
        # Generate response with image
        response = model.generate_content([
            prompt,
            {
                'mime_type': 'image/jpeg',
                'data': image_data
            }
        ])
        
        # Parse JSON response
        response_text = response.text.strip()
        logging.debug(f"Gemini raw response: {response_text}")
        
        # Clean up response (sometimes models wrap JSON in code blocks)
        if response_text.startswith('```'):
            # Remove code block markers
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Ensure all required keys exist
        extracted = {
            'brand': result.get('brand', 'Unknown'),
            'model': result.get('model', 'Unknown'),
            'serial': result.get('serial', 'Unknown'),
            'other': ''  # Gemini doesn't extract 'other' text
        }
        
        logging.info(f"Gemini extraction successful: Brand={extracted['brand']}, "
                    f"Model={extracted['model']}, Serial={extracted['serial']}")
        
        return extracted
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Gemini response as JSON: {e}")
        logging.error(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': f'Gemini parsing error: {str(e)}'
        }
    except Exception as e:
        logging.error(f"Gemini extraction failed: {type(e).__name__}: {e}", exc_info=True)
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': f'Gemini API error: {str(e)}'
        }
        return {
            'brand': 'Unknown',
            'model': 'Unknown',
            'serial': 'Unknown',
            'other': f'Gemini API error: {str(e)}'
        }


def calculate_confidence(result: Dict[str, str]) -> int:
    """
    Calculate confidence score (0-100) based on extraction quality.
    
    Args:
        result: Dictionary with 'brand', 'model', 'serial' keys
    
    Returns:
        Confidence score (0-100)
    """
    score = 0
    
    brand = result.get('brand', '')
    model = result.get('model', '')
    serial = result.get('serial', '')
    
    # Brand scoring (0-30 points)
    if brand and brand != 'Unknown':
        # Known brands get full points
        known_brands = ['GE', 'Whirlpool', 'Samsung', 'Maytag', 'Frigidaire', 
                       'LG', 'Bosch', 'KitchenAid', 'Electrolux', 'Kenmore']
        if any(kb.lower() in brand.lower() for kb in known_brands):
            score += 30
        else:
            score += 20  # Some brand detected
    
    # Model scoring (0-40 points)
    if model and model != 'Unknown':
        model_clean = model.replace(' ', '').replace('-', '')
        # Good models are 6-20 chars with mix of letters and numbers
        if 6 <= len(model_clean) <= 20:
            # Check for reasonable alphanumeric mix
            digits = sum(c.isdigit() for c in model_clean)
            letters = sum(c.isalpha() for c in model_clean)
            if digits > 0 and letters > 0:
                score += 40
            elif digits > 3 or letters > 3:
                score += 30
            else:
                score += 20
        elif 4 <= len(model_clean) <= 25:
            score += 25
        else:
            score += 10
    
    # Serial scoring (0-30 points)
    if serial and serial != 'Unknown':
        serial_clean = serial.replace(' ', '').replace('-', '')
        # Good serials are 6-20 chars, mostly alphanumeric
        if 6 <= len(serial_clean) <= 20:
            if serial_clean.isalnum():
                score += 30
            else:
                score += 20
        elif 4 <= len(serial_clean) <= 25:
            score += 15
        else:
            score += 5
    
    return min(100, max(0, score))


if __name__ == '__main__':
    # Test with a sample image
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = extract_with_gemini(test_image)
        confidence = calculate_confidence(result)
        print(f"\nExtraction Results:")
        print(f"Brand: {result['brand']}")
        print(f"Model: {result['model']}")
        print(f"Serial: {result['serial']}")
        print(f"Confidence: {confidence}%")
    else:
        print("Usage: python gemini_vision.py <image_path>")
        print("\nMake sure to set GOOGLE_API_KEY environment variable first:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
