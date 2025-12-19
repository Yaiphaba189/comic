import cv2
import numpy as np

def create_comic_layout(panels, panels_per_row=3):
    """
    Stitches list of panel images (numpy arrays) into a grid.
    """
    if not panels:
        return None
        
    num_panels = len(panels)
    
    # Assume all panels are same size for now, or resize them
    h, w, c = panels[0].shape
    
    # Calculate grid size
    rows = (num_panels + panels_per_row - 1) // panels_per_row
    
    # Canvas size (plus margins)
    margin = 20
    canvas_w = (w * panels_per_row) + (margin * (panels_per_row + 1))
    canvas_h = (h * rows) + (margin * (rows + 1))
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 # White background
    
    for i, panel in enumerate(panels):
        row = i // panels_per_row
        col = i % panels_per_row
        
        y_start = margin + (row * (h + margin))
        x_start = margin + (col * (w + margin))
        
        # Resize if needed (safety check)
        if panel.shape != (h, w, c):
             panel = cv2.resize(panel, (w, h))
             
        # Insert panel
        canvas[y_start:y_start+h, x_start:x_start+w] = panel
        
        # Draw border
        cv2.rectangle(canvas, (x_start, y_start), (x_start+w, y_start+h), (0, 0, 0), 2)
        
    return canvas
