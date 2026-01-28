#!/usr/bin/env python3
"""
Generate PWA icons with gradient background and appliance theme
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_gradient_background(size, color1, color2):
    """Create a gradient background from color1 to color2"""
    base = Image.new('RGB', (size, size), color1)
    top = Image.new('RGB', (size, size), color2)
    mask = Image.new('L', (size, size))
    mask_data = []
    for y in range(size):
        for x in range(size):
            # Diagonal gradient from top-left to bottom-right
            distance = ((x / size) + (y / size)) / 2
            mask_data.append(int(distance * 255))
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base

def create_icon(size, filename):
    """Create an app icon at the specified size"""
    # Gradient colors (purple to blue)
    color1 = (102, 126, 234)  # #667eea
    color2 = (118, 75, 162)   # #764ba2
    
    # Create gradient background
    img = create_gradient_background(size, color1, color2)
    draw = ImageDraw.Draw(img)
    
    # Add rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    corner_radius = size // 5  # 20% corner radius
    mask_draw.rounded_rectangle(
        [(0, 0), (size, size)],
        radius=corner_radius,
        fill=255
    )
    
    # Apply rounded corner mask
    output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0))
    output.putalpha(mask)
    
    # Add icon text (emoji-style)
    try:
        # Try to load a system font
        font_size = size // 2
        try:
            # Try Apple system font first (macOS)
            font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", font_size)
        except:
            try:
                # Try common macOS font
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                # Fall back to default
                font = ImageFont.load_default()
        
        # Add magnifying glass emoji
        text = "🔍"
        
        # Get text size for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]
        
        # Draw white background circle for better visibility
        circle_radius = size // 3
        circle_center = (size // 2, size // 2)
        draw.ellipse(
            [
                circle_center[0] - circle_radius,
                circle_center[1] - circle_radius,
                circle_center[0] + circle_radius,
                circle_center[1] + circle_radius
            ],
            fill=(255, 255, 255, 200)
        )
        
        # Draw the text
        draw.text((x, y), text, fill=(102, 126, 234), font=font)
        
    except Exception as e:
        print(f"Could not add text: {e}")
        # If text fails, just draw a simple icon shape
        center = size // 2
        radius = size // 3
        
        # Draw white circle background
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill='white'
        )
        
        # Draw magnifying glass shape
        glass_radius = radius // 2
        draw.ellipse(
            [center - glass_radius, center - glass_radius, 
             center + glass_radius, center + glass_radius],
            outline=(102, 126, 234),
            width=size // 30
        )
        
        # Draw handle
        handle_start_x = center + int(glass_radius * 0.7)
        handle_start_y = center + int(glass_radius * 0.7)
        handle_end_x = center + glass_radius + size // 10
        handle_end_y = center + glass_radius + size // 10
        draw.line(
            [(handle_start_x, handle_start_y), (handle_end_x, handle_end_y)],
            fill=(102, 126, 234),
            width=size // 30
        )
    
    # Save the image
    output.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

if __name__ == '__main__':
    # Change to the static directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'appliance_lookup', 'static')
    
    os.chdir(static_dir)
    
    # Create icons
    create_icon(192, 'icon-192.png')
    create_icon(512, 'icon-512.png')
    
    print("\n✓ PWA icons created successfully!")
    print("  - icon-192.png (192x192)")
    print("  - icon-512.png (512x512)")
