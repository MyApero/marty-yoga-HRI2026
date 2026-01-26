import re

# Keywords to look for in the text
BODY_PARTS = {
    "head",
    "neck",
    "shoulder",
    "arm",
    "elbow",
    "wrist",
    "hand",
    "spine",
    "back",
    "hip",
    "leg",
    "knee",
    "ankle",
    "foot",
    "toe",
}

# Map common speech terms to your specific config keys
# Config Keys: Right/Left Wrist, Elbow, Shoulder, Knee, Hip, Ankle, Spine, Spine Alignment
SYNONYMS = {
    "back": "spine",  # "Keep back straight" -> matches "Spine" or "Spine Alignment"
    "legs": "knee",  # loose mapping if needed, usually specific enough
    "arms": "elbow",  # loose mapping
}


def extract_keywords(text):
    """Simple helper to find body parts in text."""
    if not text:
        return set()
    # Normalize text: lowercase and remove punctuation
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    found = set()
    for word in normalized.split():
        # Check direct match or simple plurals
        if word in BODY_PARTS:
            found.add(word)
        elif word.endswith("s") and word[:-1] in BODY_PARTS:
            found.add(word[:-1])
    return found
