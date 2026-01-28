import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def assess_image_quality(image_path):
    """
    Assess the quality of an image for OCR purposes.
    Returns quality score (0-100) and specific issues detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "score": 0,
            "issues": ["Could not read image"],
            "recommendations": ["Upload a valid image file"]
        }
    
    issues = []
    recommendations = []
    score = 100
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 1. Check image size
    if width < 800 or height < 600:
        issues.append("Image resolution is low")
        recommendations.append("Take photo closer to the label to fill the frame")
        score -= 20
    
    # 2. Check for blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logging.debug(f"Blur score (Laplacian variance): {laplacian_var}")
    
    if laplacian_var < 50:
        issues.append("Image is very blurry")
        recommendations.append("Hold phone steady and tap to focus on the label before taking photo")
        score -= 30
    elif laplacian_var < 100:
        issues.append("Image is somewhat blurry")
        recommendations.append("Try to keep phone steady and ensure label is in focus")
        score -= 15
    
    # 3. Check for proper exposure (brightness)
    mean_brightness = np.mean(gray)
    logging.debug(f"Brightness: {mean_brightness}")
    
    if mean_brightness < 50:
        issues.append("Image is too dark")
        recommendations.append("Increase lighting or use flash")
        score -= 20
    elif mean_brightness > 200:
        issues.append("Image is overexposed (too bright)")
        recommendations.append("Reduce lighting or move away from direct light source")
        score -= 20
    
    # 4. Check for glare (bright spots)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    bright_pixels = np.sum(bright_mask == 255)
    bright_ratio = bright_pixels / (width * height)
    logging.debug(f"Bright pixel ratio (glare): {bright_ratio}")
    
    if bright_ratio > 0.3:
        issues.append("Significant glare detected")
        recommendations.append("Reduce glare by changing angle or moving away from reflective surfaces")
        score -= 25
    elif bright_ratio > 0.15:
        issues.append("Some glare detected")
        recommendations.append("Try to minimize reflections on the label")
        score -= 10
    
    # 5. Check contrast
    contrast = gray.std()
    logging.debug(f"Contrast: {contrast}")
    
    if contrast < 30:
        issues.append("Low contrast - hard to read text")
        recommendations.append("Improve lighting to increase contrast between text and background")
        score -= 15
    
    # 6. Check if image is tilted (perspective issue)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None and len(lines) > 10:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Normalize to 0-90
            if angle > 90:
                angle = 180 - angle
            angles.append(angle)
        
        # Check if most lines are significantly tilted
        mean_angle = np.mean(angles)
        logging.debug(f"Mean line angle: {mean_angle}")
        
        if mean_angle > 15 and mean_angle < 75:
            issues.append("Label appears tilted")
            recommendations.append("Hold phone parallel to the label surface")
            score -= 15
    
    # 7. Check if label fills frame
    # Use edge detection to find label boundaries
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        image_area = width * height
        fill_ratio = contour_area / image_area
        logging.debug(f"Label fill ratio: {fill_ratio}")
        
        if fill_ratio < 0.2:
            issues.append("Label is too small in frame")
            recommendations.append("Move closer to fill more of the frame with the label")
            score -= 20
        elif fill_ratio < 0.4:
            issues.append("Label could be larger in frame")
            recommendations.append("Try moving closer to the label")
            score -= 10
    
    # Ensure score doesn't go below 0
    score = max(0, score)
    
    # Add positive feedback if quality is good
    if score >= 80:
        issues = ["Image quality is good"]
        recommendations = []
    
    quality_level = "good" if score >= 70 else "fair" if score >= 40 else "poor"
    
    logging.debug(f"Image quality assessment - Score: {score}, Level: {quality_level}")
    
    return {
        "score": score,
        "level": quality_level,
        "issues": issues,
        "recommendations": recommendations
    }
