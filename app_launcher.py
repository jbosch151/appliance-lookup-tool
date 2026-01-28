import os
import sys

# Load environment variables from .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Ensure the working directory is set to the root of the project
os.environ["FLASK_RUN_FROM_CWD"] = "1"

# Add the 'appliance_lookup' folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'appliance_lookup'))

# Run the app on a different port to avoid conflict
from appliance_lookup.app import app

if __name__ == "__main__":
    # Get port from environment variable (for cloud deployment) or use 5001
    port = int(os.getenv('PORT', 5001))
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    
    print("=" * 70)
    print("Appliance Lookup Tool - Enhanced with AI Vision")
    print("=" * 70)
    
    # Check if Gemini API key is set
    if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_API_KEY') != 'your-api-key-here':
        print("✓ Gemini Vision AI: ENABLED (Primary extraction method)")
        print("  Using Google's multimodal AI for superior accuracy")
    else:
        print("⚠ Gemini Vision AI: NOT CONFIGURED")
        print("  Will use traditional OCR (lower accuracy)")
        print("  To enable: Get free API key at https://makersuite.google.com/app/apikey")
        print("  Then edit .env file: GOOGLE_API_KEY=your-key-here")
    
    print("\n✓ Fallback OCR: Tesseract + EasyOCR available if needed")
    print("=" * 70)
    
    if debug_mode:
        print("\nServer running on:")
        print("  - Local:  http://127.0.0.1:5001")
        print("  - Network: http://192.168.1.32:5001 (for mobile devices)")
        print("\nPress CTRL+C to stop\n")
    else:
        print(f"\nProduction server starting on port {port}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
