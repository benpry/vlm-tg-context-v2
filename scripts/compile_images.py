#!/usr/bin/env python3
"""
Script to compile SVG images into a single grid with labels.
Reads page-A.svg through page-L.svg and creates a 3x4 grid with labels A-L.
"""

import os
from io import BytesIO

import cairosvg
from PIL import Image, ImageDraw, ImageFont
from pyprojroot import here


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
    label_height = 60
    padding = 20

    # Get the directory containing the images
    images_dir = here("data/images")

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

    # Calculate cell and grid dimensions
    cell_width = img_width + padding
    cell_height = img_height + label_height
    total_width = grid_cols * cell_width + (grid_cols + 1) * padding
    total_height = grid_rows * cell_height + (grid_rows + 1) * padding

    print(f"Grid dimensions: {total_width}x{total_height}")

    # Create the composite image with light gray background
    composite = Image.new("RGB", (total_width, total_height), "#f0f0f0")
    draw = ImageDraw.Draw(composite)

    # Try common system fonts first, then fall back to PIL's default font.
    font = None
    for font_path in (
        "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, size=36)
            print(f"Using font {font_path}")
            break
        except OSError:
            continue

    # Place images and labels in grid
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // grid_cols
        col = i % grid_cols

        # Calculate cell position (with gutter padding around all edges)
        cell_x = padding + col * (cell_width + padding)
        cell_y = padding + row * (cell_height + padding)

        # Draw white card background for the entire cell
        draw.rectangle(
            [cell_x, cell_y, cell_x + cell_width, cell_y + cell_height],
            fill="white",
        )

        # Paste the image centered horizontally in the cell
        img_x = cell_x + (cell_width - img_width) // 2
        img_y = cell_y + padding // 2
        composite.paste(img, (img_x, img_y))

        # Add the label centered below the image
        label_y = img_y + img_height + 12
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = cell_x + (cell_width - text_width) // 2

        draw.text((text_x, label_y), label, fill="red", font=font)

        print(f"Placed image {label} at position ({cell_x}, {cell_y})")

    # Save the composite image
    output_path = here("data/compiled_grid.png")
    composite.save(output_path, quality=95)
    print(f"Grid saved as {output_path}")

    return composite


if __name__ == "__main__":
    create_image_grid()
