
"""
import clip as clip_library

def available_models():
    return clip_library.available_models()

def load_model(model_name):
    return clip_library.load(model_name)

# Add any other functions you need
"""
import clip
print(clip.__file__)  # This will show you which clip module is being imported