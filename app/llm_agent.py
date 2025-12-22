import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
import PIL.Image

current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env.example'
load_result = load_dotenv(dotenv_path=".env.example")

print(f"Loading .env from: {env_path}")
print(f"Did it load? {load_result}")


class LLMAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Flash is faster for real-time

    def analyze_image(self, image_array, prompt="What is this object?"):
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_image)

        try:
            response = self.model.generate_content([prompt, pil_image])
            return response.text
        except Exception as e:
            return f"Error connecting to Gemini: {e}"


# --- DEBUG/TEST BLOCK ---
if __name__ == "__main__":
    import numpy as np

    print("Testing LLM Agent...")
    try:
        agent = LLMAgent()
        print("LLM Agent initialized. Sending test image...")

        # Create a dummy blank white image (100x100 pixels)
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Ask a simple question
        response = agent.analyze_image(dummy_image, prompt="What color is this image? Reply with one word.")
        print(f"GEMINI RESPONSE: {response}")
        print("Test Passed!")
    except Exception as e:
        print(f"\nTest Failed. Check your API Key in .env.\nError: {e}")