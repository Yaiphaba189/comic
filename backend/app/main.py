from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import os
import uuid

# Import modules
from .emotion_model.inference import emotion_predictor
from .nlp.story_parser import parse_story
from .nlp.visual_prompt import generate_visual_instruction
from .creative_cv.sd_generator import generate_panel_sd
from .vision.layout_engine import create_comic_layout
from .vision.text_renderer import add_caption

app = FastAPI(title="AI Comic Strip Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup output dir
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StoryRequest(BaseModel):
    story_text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Comic Strip Generator API (CV Edition)"}

@app.post("/generate")
def generate_comic(request: StoryRequest):
    story_text = request.story_text
    if not story_text:
        raise HTTPException(status_code=400, detail="Story text cannot be empty")
        
    # 1. Parse Story
    panels_data = parse_story(story_text)
    if not panels_data:
        raise HTTPException(status_code=400, detail="Could not extract panels from story")
    
    generated_panels = []
    
    # 2. Process each panel
    for p in panels_data:
        text = p["text"]
        
        # 3. Detect Emotion
        try:
            emotion = emotion_predictor.predict(text)
        except Exception as e:
            print(f"Emotion prediction failed: {e}")
            emotion = "neutral"
            
        print(f"Panel {p['id']}: Emotion='{emotion}' | Text='{text}'")
        
        # 4. Generate Visual Instructions
        visual_instr = generate_visual_instruction(p, emotion)
        
        # 5. Render Panel Image (Stable Diffusion)
        # 512x512 resolution per panel
        prompt = visual_instr["prompt"]
        negative_prompt = visual_instr["negative_prompt"]
        
        print(f"Generating with SD: {prompt}")
        
        pil_image = generate_panel_sd(prompt, negative_prompt, width=512, height=512)
        
        if pil_image is None:
             # Fallback to white noise or blank if SD fails
             panel_img = np.zeros((512, 512, 3), np.uint8) + 255
        else:
             # Convert PIL RGB to OpenCV BGR
             panel_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 6. Add Text Overlay
        panel_img = add_caption(panel_img, text)
        
        generated_panels.append(panel_img)
        
    # 7. Assembled Comic
    comic_strip = create_comic_layout(generated_panels, panels_per_row=3)
    
    if comic_strip is None:
        raise HTTPException(status_code=500, detail="Failed to generate comic strip")
        
    # Save to disk
    filename = f"comic_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, comic_strip)
    
    # Return URL or Base64 (simplifying to returning file path/url for now)
    # In a real app, serve static files or upload to S3.
    # We will return a static URL or base64 for the frontend to consume easily for this demo.
    
    # Let's return local path for debug, and base64 for direct display
    _, buffer = cv2.imencode('.jpg', comic_strip)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": "success",
        "filepath": filepath,
        "image_base64": img_base64,
        "panels_generated": len(generated_panels)
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
