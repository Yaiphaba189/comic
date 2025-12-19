import cv2
import numpy as np
from .shapes import draw_smooth_blob, draw_jagged_star, draw_droplets, draw_shaky_lines, draw_smooth_blob as draw_explosion # Reuse blob for now or add new

def render_panel(width, height, visual_instr):
    """
    Renders a single panel based on visual instruction.
    Returns: PIL Image (for compatibility) or numpy array
    """
    # Create background
    bg_color = visual_instr.get("bg_color", (255, 255, 255))
    # OpenCV uses BGR
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])
    
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = bg_color_bgr
    
    # Draw Main Shape
    shape_type = visual_instr.get("shape_type", "circle")
    shape_color = visual_instr.get("shape_color", (0, 0, 0))
    shape_color_bgr = (shape_color[2], shape_color[1], shape_color[0])
    
    center = (width // 2, height // 2)
    size = min(width, height) // 3
    
    if shape_type == "smooth_blob":
        draw_smooth_blob(img, center, shape_color_bgr, size)
    elif shape_type == "jagged_star":
        draw_jagged_star(img, center, shape_color_bgr, size)
    elif shape_type == "droplet":
        draw_droplets(img, center, shape_color_bgr, size)
    elif shape_type == "shaky_lines":
        draw_shaky_lines(img, center, shape_color_bgr, size)
    elif shape_type == "explosion":
        draw_jagged_star(img, center, shape_color_bgr, size * 1.5) # Bigger star
    else:
        # Default circle
        cv2.circle(img, center, size, shape_color_bgr, -1)
        
    return img
