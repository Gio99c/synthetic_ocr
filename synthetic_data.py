import os
import random
import string
import json
import argparse
from math import sin, cos
from multiprocessing import Pool
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

# --------------------------
# Global Constants
# --------------------------
OUTPUT_DIR = "synthetic_dataset"
IMAGE_SIZE = (512, 512)
FONTS = [
    "fonts/arial.ttf",
    "fonts/times.ttf",
    "fonts/courier.ttf",
    "fonts/helvetica.ttf"
]
NUM_SAMPLES = 1000
MAX_TEXT_LENGTH = 15
TEXT_LENGTH_MEAN = 3
TEXT_LENGTH_STD = 2

# Text size control
TEXT_SIZE_RANGE = (100, 180)

# Probability of having an underline
UNDERLINE_PROB = 0.5

# Slightly smaller padding so shape doesn't overshadow text
SYMBOL_PADDING = 0.1

UNDERLINE_OFFSET = 0.07
BACKGROUND_COLOR = (240, 240, 240)

# Allowed discrete rotations
ROTATIONS = [0, 90, 270]

# Floor-plan style patterns
PATTERNS = ["grid", "dots", "crosses", "diagonal", "none"]

# Text generation
SEPARATORS = ['-', '_', '.', ' ', '/', '|', ':', ',', "'", '"']
ALPHANUMERIC = string.ascii_uppercase + string.digits
VALID_CHARS = ALPHANUMERIC + ''.join(SEPARATORS)

# Shapes we allow
SHAPES = [
    "rectangle", "square", "circle", "triangle",
    "rhombus", "hexagon", "half_circle", "none"
]

# Minimum shape dimension to avoid overly thin shapes
MIN_SHAPE_DIM = 40

# A known fallback font that should exist in your 'fonts' folder:
FALLBACK_FONT = "fonts/arial.ttf"

# --------------------------
# Shape & Pattern Utilities
# --------------------------

def shape_polygon(shape_type, w, h):
    """Return (x,y) points that define the shape in a w x h bounding box."""
    if shape_type == "none":
        return []
    if shape_type == "rectangle":
        return [(0,0),(w,0),(w,h),(0,h)]
    elif shape_type == "square":
        side = min(w,h)
        return [(0,0),(side,0),(side,side),(0,side)]
    elif shape_type == "triangle":
        return [(w//2, 0), (0, h), (w, h)]
    elif shape_type == "circle":
        # approximate circle with polygon
        pts = []
        r = min(w,h)//2
        cx, cy = w//2, h//2
        steps = 36
        for i in range(steps):
            ang = 2.0 * 3.14159 * i/steps
            px = cx + int(r*cos(ang))
            py = cy + int(r*sin(ang))
            pts.append((px,py))
        return pts
    elif shape_type == "rhombus":
        return [(w//2,0),(w,h//2),(w//2,h),(0,h//2)]
    elif shape_type == "hexagon":
        cx, cy = w//2, h//2
        r = min(w,h)//2
        pts = []
        for i in range(6):
            ang = (3.14159/3.0) * i
            px = cx + int(r*cos(ang))
            py = cy + int(r*sin(ang))
            pts.append((px,py))
        return pts
    elif shape_type == "half_circle":
        # top half circle
        cx, cy = w//2, h//2
        r = min(w,h)//2
        steps = 18
        pts = []
        for i in range(steps+1):
            ang = 3.14159 * i/steps
            px = cx + int(r*cos(ang))
            py = cy - int(r*sin(ang))  # top half
            pts.append((px,py))
        pts.append((cx+r,cy))
        pts.append((cx-r,cy))
        return pts
    return []

def apply_pattern(draw, bbox, pattern_type=None):
    """
    Draw pattern lines inside 'bbox' on the 'draw' surface:
    - "grid", "dots", "crosses", "diagonal", or "none".
    """
    if pattern_type is None:
        pattern_type = random.choice(PATTERNS)
    if pattern_type == "none":
        return
    x1, y1, x2, y2 = bbox

    def pattern_crosses(bx1, by1, bx2, by2, step=20):
        for xx in range(bx1, bx2, step):
            for yy in range(by1, by2, step):
                draw.line([(xx, yy-2), (xx, yy+2)], fill=(200, 200, 200), width=1)
                draw.line([(xx-2, yy), (xx+2, yy)], fill=(200, 200, 200), width=1)

    def pattern_diagonal(bx1, by1, bx2, by2, step=20):
        width = bx2 - bx1
        height = by2 - by1
        for offset in range(-max(width, height), max(width, height), step):
            p1 = (bx1, by1 + offset)
            p2 = (bx2, by2 + offset)
            draw.line([p1, p2], fill=(200, 200, 200), width=1)

    if pattern_type == "grid":
        step = 20
        for xx in range(x1, x2, step):
            draw.line([(xx, y1), (xx, y2)], fill=(200, 200, 200), width=1)
        for yy in range(y1, y2, step):
            draw.line([(x1, yy), (x2, yy)], fill=(200, 200, 200), width=1)

    elif pattern_type == "dots":
        step = 20
        for xx in range(x1, x2, step):
            for yy in range(y1, y2, step):
                draw.ellipse([(xx-2, yy-2), (xx+2, yy+2)], fill=(200,200,200))

    elif pattern_type == "crosses":
        pattern_crosses(x1, y1, x2, y2)

    elif pattern_type == "diagonal":
        pattern_diagonal(x1, y1, x2, y2)

def draw_shape_polygon(w, h):
    """
    Create an RGBA surface for the background shape
    and fill + pattern it via polygon. 
    """
    shape_surf = Image.new('RGBA', (w, h), (0,0,0,0))
    return shape_surf, ImageDraw.Draw(shape_surf)

def fill_polygon_with_pattern(shape_draw, w, h, shape_type):
    """
    Fills the shape's polygon with white, then draws pattern inside it 
    by masking with the same polygon. Then draws a black border.
    """
    pts = shape_polygon(shape_type, w, h)
    if not pts:
        return None  # shape_type might be "none"

    # Fill shape
    shape_draw.polygon(pts, fill=(255,255,255,255))

    # Pattern
    pattern_surf = Image.new('RGBA', (w, h), (0,0,0,0))
    pat_draw = ImageDraw.Draw(pattern_surf)
    apply_pattern(pat_draw, (0,0, w, h), random.choice(["dots","crosses","diagonal","none"]))

    # Mask
    mask = Image.new('L', (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon(pts, fill=255)
    
    # Paste pattern
    return pattern_surf, mask, pts


# --------------------------
# Generation Steps
# --------------------------

def generate_text():
    """Generate text with symbols acting as separators between alphanumeric segments."""
    def generate_segment(min_len=1, max_len=3):
        """Generate a short alphanumeric segment."""
        return ''.join(random.choices(ALPHANUMERIC, k=random.randint(min_len, max_len)))

    # Generate base length using normal distribution
    base_length = max(1, min(MAX_TEXT_LENGTH, 
                          int(random.gauss(TEXT_LENGTH_MEAN, TEXT_LENGTH_STD))))
    
    # Determine number of segments (1-3) based on length
    num_segments = min(base_length, 3)
    if base_length > 4:
        num_segments = random.randint(1, 3)
    
    # Generate segments and separators
    segments = [generate_segment(1, 3) for _ in range(num_segments)]
    separators = random.choices(SEPARATORS, k=num_segments-1)
    
    # Combine with separators
    text = segments[0]
    for i, sep in enumerate(separators):
        text += sep + segments[i+1]
    
    # Trim to desired length if needed
    text = text[:base_length]
    
    # Ensure at least one alphanumeric character remains
    if not any(c in ALPHANUMERIC for c in text):
        text = generate_segment(1, 3) + text[1:]
    
    return text


def generate_background(draw, img):
    """
    Fills entire background with grid or none,
    then places up to 3 random shapes, each built as a polygon with pattern.
    """
    if random.choice(["grid","none"]) == "grid":
        step = 30
        for x in range(0, IMAGE_SIZE[0], step):
            draw.line([(x, 0), (x, IMAGE_SIZE[1])], fill=(210,210,210), width=1)
        for y in range(0, IMAGE_SIZE[1], step):
            draw.line([(0, y), (IMAGE_SIZE[0], y)], fill=(210,210,210), width=1)

    shape_count = random.randint(0,3)
    for _ in range(shape_count):
        w = random.randint(MIN_SHAPE_DIM, IMAGE_SIZE[0]//2)
        h = random.randint(MIN_SHAPE_DIM, IMAGE_SIZE[1]//2)
        cx = random.randint(w//2, IMAGE_SIZE[0]-w//2)
        cy = random.randint(h//2, IMAGE_SIZE[1]-h//2)

        shape_type = random.choice(["rectangle","triangle","circle"]) 

        shape_surf, shape_draw = draw_shape_polygon(w,h)
        
        # Ensure polygon points are correct
        pts = shape_polygon(shape_type, w, h)
        if not pts:
            continue  # Skip if no valid points

        # Fill shape with white
        shape_draw.polygon(pts, fill=(255,255,255,255))

        # Pattern
        pattern_surf = Image.new('RGBA', (w, h), (0,0,0,0))
        pat_draw = ImageDraw.Draw(pattern_surf)
        apply_pattern(pat_draw, (0,0, w, h), random.choice(["dots","crosses","diagonal","none"]))

        # Mask
        mask = Image.new('L', (w, h), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.polygon(pts, fill=255)
        
        # Paste patterned surface
        shape_surf.paste(pattern_surf, (0,0), mask)

        # Explicitly draw border with more robust method
        for i in range(len(pts)):
            start = pts[i]
            end = pts[(i+1) % len(pts)]
            shape_draw.line([start, end], fill=(0,0,0), width=3)

        px = cx - w//2
        py = cy - h//2
        img.paste(shape_surf, (px, py), shape_surf)


def create_text_and_underline_surface(text, font):
    """
    Create an RGBA surface with text + optional underline.
    """
    text_w, text_h = font.getbbox(text)[2:]
    offset_pixels = int(text_h * UNDERLINE_OFFSET)
    surf_w = text_w
    surf_h = text_h + offset_pixels + 4
    text_surf = Image.new('RGBA', (surf_w, surf_h), (0,0,0,0))
    tdraw = ImageDraw.Draw(text_surf)
    tdraw.text((0,0), text, font=font, fill=(0,0,0,255))

    # Conditional underline with proportional thickness
    if random.random() < UNDERLINE_PROB:
        underline_thickness = max(1, int(text_h * 0.08))  # 8% of text height
        underline_y = text_h + offset_pixels
        tdraw.line([(0,underline_y),(text_w,underline_y)], 
                 fill=(0,0,0), width=underline_thickness)
    
    return text_surf, text_w, text_h


def draw_shape_and_pattern(shape_type, shape_w, shape_h):
    """
    Create a shape + pattern in a bigger canvas to avoid pattern bleed on rotation.
    Returns (shape_surf, margin).
    """
    margin = max(shape_w, shape_h)
    canvas_w = shape_w + 2*margin
    canvas_h = shape_h + 2*margin
    shape_surf = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    shape_draw = ImageDraw.Draw(shape_surf)

    if shape_type == "none":
        return shape_surf, margin

    pts = shape_polygon(shape_type, shape_w, shape_h)
    if not pts:
        return shape_surf, margin

    offset_pts = [(x+margin,y+margin) for (x,y) in pts]

    # fill shape
    shape_draw.polygon(offset_pts, fill=(255,255,255,255))

    # pattern
    pat_surf = Image.new('RGBA',(canvas_w, canvas_h),(0,0,0,0))
    pat_draw = ImageDraw.Draw(pat_surf)
    apply_pattern(pat_draw,(0,0,canvas_w,canvas_h), random.choice(["dots","crosses","diagonal","none"]))

    mask = Image.new('L',(canvas_w,canvas_h),0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(offset_pts,fill=255)
    shape_surf.paste(pat_surf,(0,0),mask)

    # More robust border drawing
    for i in range(len(offset_pts)):
        start = offset_pts[i]
        end = offset_pts[(i+1) % len(offset_pts)]
        shape_draw.line([start, end], fill=(0,0,0), width=3)

    return shape_surf, margin


def clamp(val, low, high):
    return max(low, min(high, val))

def generate_synthetic_sample(idx, output_dir):
    # 1) Create image + background
    img = Image.new('RGB', IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    generate_background(draw, img)

    # 2) Generate text
    text = generate_text()

    # 3) Pick font or fallback
    try:
        initial_font_size = random.randint(*TEXT_SIZE_RANGE)
        font_path = random.choice(FONTS)
        font = ImageFont.truetype(font_path, initial_font_size)
    except OSError:
        font_path = FALLBACK_FONT
        initial_font_size = random.randint(*TEXT_SIZE_RANGE)
        try:
            font = ImageFont.truetype(font_path, initial_font_size)
        except OSError:
            font = ImageFont.load_default()

    # 4) First measure text to get initial dimensions
    text_w, text_h = font.getbbox(text)[2:]
    angle = random.choice(ROTATIONS)
    
    # Calculate rotated text dimensions - not used now
    if angle in [90, 270]:
        rotated_text_w, rotated_text_h = text_h, text_w
    else:
        rotated_text_w, rotated_text_h = text_w, text_h

    # Calculate shape size with padding
    shape_w = int(max(text_w, text_h) * (1 + SYMBOL_PADDING))
    shape_h = shape_w
    shape_type = random.choice(SHAPES)

    # 5) Create shape surface
    shape_surf, margin = draw_shape_and_pattern(shape_type, shape_w, shape_h)
    rotated_shape = shape_surf.rotate(angle, expand=True, resample=Image.NEAREST)
    rs_w, rs_h = rotated_shape.size

    # Check if shape is too big for image and scale down if necessary
    scale_factor = 1.0
    if rs_w >= IMAGE_SIZE[0] or rs_h >= IMAGE_SIZE[1]:
        scale_factor = min(IMAGE_SIZE[0] / rs_w, IMAGE_SIZE[1] / rs_h) * 0.95  # 5% margin
        new_font_size = int(initial_font_size * scale_factor)
        font = ImageFont.truetype(font_path, new_font_size)
        
        # Recalculate everything with new size
        text_w, text_h = font.getbbox(text)[2:]
        shape_w = int(max(text_w, text_h) * (1 + SYMBOL_PADDING))
        shape_h = shape_w
        
        shape_surf, margin = draw_shape_and_pattern(shape_type, shape_w, shape_h)
        rotated_shape = shape_surf.rotate(angle, expand=True, resample=Image.NEAREST)
        rs_w, rs_h = rotated_shape.size

    # Calculate valid center positions
    center_x = random.randint(rs_w//2, IMAGE_SIZE[0]-rs_w//2) if rs_w < IMAGE_SIZE[0] else IMAGE_SIZE[0]//2
    center_y = random.randint(rs_h//2, IMAGE_SIZE[1]-rs_h//2) if rs_h < IMAGE_SIZE[1] else IMAGE_SIZE[1]//2

    # Position shape
    sx = clamp(center_x - rs_w//2, 0, IMAGE_SIZE[0]-rs_w)
    sy = clamp(center_y - rs_h//2, 0, IMAGE_SIZE[1]-rs_h)
    img.paste(rotated_shape, (sx,sy), rotated_shape)

    # 6) Create text surface with updated font
    text_surf, real_text_w, real_text_h = create_text_and_underline_surface(text, font)
    rotated_text = text_surf.rotate(angle, expand=True, resample=Image.NEAREST)
    rt_w, rt_h = rotated_text.size

    # Position text in center of shape
    tx = clamp(center_x - rt_w//2, 0, IMAGE_SIZE[0]-rt_w)
    ty = clamp(center_y - rt_h//2, 0, IMAGE_SIZE[1]-rt_h)
    
    # Final safety check - if still out of bounds, skip text
    if tx + rt_w > IMAGE_SIZE[0] or ty + rt_h > IMAGE_SIZE[1]:
        return generate_synthetic_sample(idx, output_dir)
        
    img.paste(rotated_text, (tx,ty), rotated_text)
    text_bbox = (tx, ty, tx+rt_w, ty+rt_h)

    # 7) Save
    img_path = os.path.join(output_dir, f"sample_{idx}.png")
    img.save(img_path)
    annotation = {
        "image": img_path,
        "bbox": text_bbox,
        "text": text,
        "rotation": angle,
        "shape": shape_type
    }
    with open(os.path.join(output_dir, f"sample_{idx}.json"), "w") as f:
        json.dump(annotation, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to store images/annotations.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel processes.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.workers>1:
        with Pool(args.workers) as p:
            p.starmap(generate_synthetic_sample, [(i,args.output_dir) for i in range(args.num_samples)])
    else:
        for i in tqdm(range(args.num_samples), desc="Generating"):
            generate_synthetic_sample(i, args.output_dir)

    print(f"\nDone! Generated {args.num_samples} samples in '{args.output_dir}'")

if __name__=="__main__":
    main()
