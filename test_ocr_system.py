#!/usr/bin/env python3
"""
Test script for the 2-stage OCR system.
Tests both PaddleOCR (Stage 1) and Gemini Vision (Stage 2).
"""
import sys
import os

# Add appliance_lookup to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'appliance_lookup'))

def test_paddle_only(image_path):
    """Test Stage 1 (PaddleOCR) only"""
    print("\n" + "="*60)
    print("TESTING STAGE 1: PaddleOCR")
    print("="*60)
    
    from paddle_ocr import extract_with_paddle
    
    result, confidence = extract_with_paddle(image_path)
    
    print(f"\n📊 Results:")
    print(f"  Brand:      {result['brand']}")
    print(f"  Model:      {result['model']}")
    print(f"  Serial:     {result['serial']}")
    print(f"  Confidence: {confidence}%")
    
    if confidence >= 75:
        print(f"\n✅ HIGH CONFIDENCE - Would use PaddleOCR result")
    else:
        print(f"\n⚠️  LOW CONFIDENCE - Would trigger Gemini fallback")
    
    return result, confidence


def test_gemini_only(image_path):
    """Test Stage 2 (Gemini Vision) only"""
    print("\n" + "="*60)
    print("TESTING STAGE 2: Gemini Vision")
    print("="*60)
    
    from gemini_vision import extract_with_gemini
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY not set!")
        print("\nTo set it:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\nGet a free key at: https://makersuite.google.com/app/apikey")
        return None
    
    result = extract_with_gemini(image_path)
    
    print(f"\n📊 Results:")
    print(f"  Brand:  {result['brand']}")
    print(f"  Model:  {result['model']}")
    print(f"  Serial: {result['serial']}")
    
    return result


def test_2stage_system(image_path):
    """Test the full 2-stage system"""
    print("\n" + "="*60)
    print("TESTING FULL 2-STAGE SYSTEM")
    print("="*60)
    
    from paddle_ocr import extract_with_paddle
    from gemini_vision import extract_with_gemini
    
    THRESHOLD = 75
    
    # Stage 1
    print("\n🔍 Stage 1: Trying PaddleOCR...")
    paddle_result, confidence = extract_with_paddle(image_path)
    print(f"   Confidence: {confidence}%")
    
    if confidence >= THRESHOLD:
        print(f"\n✅ Stage 1 SUCCESS (confidence {confidence}%)")
        print("   Using PaddleOCR result")
        final_result = paddle_result
        method = f"PaddleOCR (confidence: {confidence}%)"
    else:
        print(f"\n⚠️  Stage 1 low confidence ({confidence}%)")
        print("   Triggering Gemini fallback...")
        
        gemini_result = test_gemini_only(image_path)
        if gemini_result:
            print("\n✅ Stage 2 SUCCESS")
            print("   Using Gemini Vision result")
            final_result = gemini_result
            method = f"Gemini Vision (PaddleOCR was {confidence}%)"
        else:
            print("\n⚠️  Gemini unavailable, using PaddleOCR result")
            final_result = paddle_result
            method = f"PaddleOCR (low confidence: {confidence}%)"
    
    print(f"\n" + "="*60)
    print(f"FINAL RESULTS ({method})")
    print("="*60)
    print(f"  Brand:  {final_result['brand']}")
    print(f"  Model:  {final_result['model']}")
    print(f"  Serial: {final_result['serial']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_ocr_system.py <image_path> [--paddle-only|--gemini-only|--full]")
        print("\nExamples:")
        print("  python test_ocr_system.py image.jpg")
        print("  python test_ocr_system.py image.jpg --paddle-only")
        print("  python test_ocr_system.py image.jpg --gemini-only")
        print("  python test_ocr_system.py image.jpg --full")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else '--full'
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\n📸 Testing with image: {image_path}")
    
    if mode == '--paddle-only':
        test_paddle_only(image_path)
    elif mode == '--gemini-only':
        test_gemini_only(image_path)
    else:
        test_2stage_system(image_path)
