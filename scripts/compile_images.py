#!/usr/bin/env python3
"""
Script to compile SVG images into a single grid with labels.
Reads page-A.svg through page-L.svg and creates a 3x4 grid with labels A-L.
"""

import os
from io import BytesIO

import cairosvg
from PIL import Image, ImageDraw, ImageFont


def svg_to_pil(svg_path):
    """Convert SVG file to PIL Image with proper background handling."""
    try:
        # Convert SVG to PNG bytes with white background
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            background_color="white",
            output_width=200,  # Set a consistent width
            output_height=200,  # Set a consistent height
        )
        # Create PIL Image from bytes
        img = Image.open(BytesIO(png_bytes))
        # Convert to RGB if necessary (removes alpha channel)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        # Return a placeholder image if conversion fails
        placeholder = Image.new("RGB", (200, 200), "lightgray")
        draw = ImageDraw.Draw(placeholder)
        draw.text((50, 90), "Error", fill="black")
        return placeholder


def create_image_grid():
    """Create a grid of images with labels A through L."""
    # Configuration
    grid_cols = 4
    grid_rows = 3
    label_height = 50
    padding = 20

    # Get the directory containing the images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "images")

    # Load all images
    images = []
    labels = []

    for letter in "ABCDEFGHIJKL":
        svg_path = os.path.join(images_dir, f"page-{letter}.svg")
        if os.path.exists(svg_path):
            print(f"Loading {svg_path}...")
            img = svg_to_pil(svg_path)
            images.append(img)
            labels.append(letter)
            print(f"Successfully loaded {svg_path} - size: {img.size}")
        else:
            print(f"Warning: {svg_path} not found")

    if not images:
        print("No images found!")
        return

    print(f"Loaded {len(images)} images")

    # Get image dimensions (all should be 200x200 now)
    img_width, img_height = images[0].size
    print(f"Image dimensions: {img_width}x{img_height}")

    # Calculate grid dimensions
    total_width = grid_cols * img_width + (grid_cols - 1) * padding
    total_height = grid_rows * (img_height + label_height) + (grid_rows - 1) * padding

    print(f"Grid dimensions: {total_width}x{total_height}")

    # Create the composite image with light gray background
    composite = Image.new("RGB", (total_width, total_height), "#f0f0f0")
    draw = ImageDraw.Draw(composite)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

    # Place images and labels in grid
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // grid_cols
        col = i % grid_cols

        # Calculate position
        x = col * (img_width + padding)
        y = row * (img_height + label_height + padding)

        # Add a white border around each image
        border_padding = 2
        draw.rectangle(
            [
                x - border_padding,
                y - border_padding,
                x + img_width + border_padding,
                y + img_height + border_padding,
            ],
            fill="white",
            outline="gray",
        )

        # Paste the image
        composite.paste(img, (x, y))

        # Add the label below the image
        label_y = y + img_height + 8

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (img_width - text_width) // 2

        # Add a white background for the text
        text_bg_padding = 4
        draw.rectangle(
            [
                text_x - text_bg_padding,
                label_y - text_bg_padding,
                text_x + text_width + text_bg_padding,
                label_y + bbox[3] - bbox[1] + text_bg_padding,
            ],
            fill="white",
            outline="gray",
        )

        draw.text((text_x, label_y), label, fill="black", font=font)

        print(f"Placed image {label} at position ({x}, {y})")

    # Save the composite image
    output_path = os.path.join(script_dir, "compiled_grid.png")
    composite.save(output_path, quality=95)
    print(f"Grid saved as {output_path}")

    return composite


if __name__ == "__main__":
    create_image_grid()
