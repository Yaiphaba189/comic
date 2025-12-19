import cv2
import numpy as np

def add_caption(image, text, position="bottom"):
    """
    Adds a caption bar to the image panel.
    """
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 0, 0)
    
    # Text wrapping logic (simple)
    max_width = w - 20
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        (text_w, _), _ = cv2.getTextSize(" ".join(current_line), font, font_scale, thickness)
        if text_w > max_width:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))
    
    # Bar height
    text_line_h = 25
    bar_h = (len(lines) * text_line_h) + 10
    
    if position == "bottom":
        # Draw white rectangle at bottom
        cv2.rectangle(image, (0, h - bar_h), (w, h), (255, 255, 255), -1)
        cv2.rectangle(image, (0, h - bar_h), (w, h), (0, 0, 0), 1) # Border
        
        y_text = h - bar_h + 20
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x_text = (w - tw) // 2
            cv2.putText(image, line, (x_text, y_text), font, font_scale, color, thickness, cv2.LINE_AA)
            y_text += text_line_h
            
    return image
