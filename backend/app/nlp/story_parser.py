import re

def parse_story(story_text):
    """
    Parses story text into panels using Regex instead of Spacy.
    """
    # Simple regex to split by . ? ! followed by space or newline
    # This is less robust than Spacy but works for standard punctuation
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    
    sentences = sentence_endings.split(story_text.strip())
    
    panels = []
    
    for i, text in enumerate(sentences):
        text = text.strip()
        if not text:
            continue
            
        # Simplistic character detection: Capitalized words that are not starting the sentence
        # (Very basic heuristic, skipping for now to avoid complexity/noise)
        chars = [] 
        
        panels.append({
            "id": i,
            "text": text,
            "characters": chars
        })
        
    return panels
