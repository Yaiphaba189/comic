import cv2
import numpy as np
import random

def draw_smooth_blob(img, center, color, size):
    """Draws a random smooth blob (circles approximation)"""
    x, y = center
    num_circles = random.randint(3, 6)
    
    cv2.circle(img, (int(x), int(y)), int(size), color, -1)
    
    for _ in range(num_circles):
        offset_x = random.randint(-int(size/2), int(size/2))
        offset_y = random.randint(-int(size/2), int(size/2))
        sub_size = random.randint(int(size/2), int(size))
        cv2.circle(img, (int(x + offset_x), int(y + offset_y)), int(sub_size), color, -1)

def draw_jagged_star(img, center, color, size):
    """Draws a jagged star/burst shape"""
    x, y = center
    pts = []
    num_points = 12
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points)
        r = size if i % 2 == 0 else size * 0.4
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        pts.append([px, py])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)

def draw_droplets(img, center, color, size):
    """Draws rain-like droplets"""
    x, y = center
    for _ in range(5):
        off_x = random.randint(-int(size), int(size))
        off_y = random.randint(-int(size), int(size))
        cv2.ellipse(img, (int(x+off_x), int(y+off_y)), (int(size/5), int(size/2)), 0, 0, 360, color, -1)

def draw_shaky_lines(img, center, color, size):
    """Draws chaotic shaky lines"""
    x, y = center
    for _ in range(10):
        x1 = x + random.randint(-int(size), int(size))
        y1 = y + random.randint(-int(size), int(size))
        x2 = x + random.randint(-int(size), int(size))
        y2 = y + random.randint(-int(size), int(size))
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
