def generate_visual_instruction(panel_data, emotion):
    """
    Maps panel content and emotion to a Stable Diffusion prompt.
    """
    
    # Base style for "Pencil Art"
    style_prompt = "pencil art sketch, graphite charcoal style, rough sketch, monochrome, highly detailed, comic book panel, character focus"
    
    # Emotion nuances
    emotion_modifiers = {
        "happy": "joyful atmosphere, smiling characters, bright lighting, soft strokes",
        "sad": "melancholic atmosphere, rain, shadows, lonely, heavy strokes",
        "angry": "intense atmosphere, sharp lines, aggressive shading, dramatic lighting, furious expression",
        "fear": "eerie atmosphere, dark shadows, trembling lines, horror style",
        "surprise": "shocked expression, dynamic composition, dramatic angle",
        "neutral": "calm atmosphere, everyday slice of life"
    }
    
    emotion_text = emotion_modifiers.get(emotion, "calm atmosphere")
    
    # Construct the full positive prompt
    # We combine the panel text (action/scene), the emotion, and the hardcoded style.
    prompt = f"{panel_data.get('text', '')}, {emotion_text}, {style_prompt}"
    
    # Negative prompt to avoid bad quality
    negative_prompt = "photorealistic, color, 3d render, plastic, deformed, blurry, bad anatomy, bad eyes, crossed eyes, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, mutated hands and fingers, out of frame, blender, doll, cropped, low-res, close-up, poorly-drawn face, out of frame duplicate, two heads, blurred, ugly, disfigured, too many fingers, deformed, repetitive, black and white, grainy, extra limbs, bad anatomy, high contrast, over saturated, glossy, cartoon, 3d, 3d render, photoshop, artistic, painting"

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }
