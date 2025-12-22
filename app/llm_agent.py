import os
from google import genai  # <--- NEW IMPORT
from google.genai import types
from dotenv import load_dotenv
from collections import Counter

# Load API Key
load_dotenv()


class LLMAgent:
    def __init__(self):
        # 1. Setup Client (New Syntax)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API Key missing! Check your .env file.")

        # The new library uses a 'Client' object
        self.client = genai.Client(api_key=api_key)

        # Use 'gemini-2.0-flash' or 'gemini-1.5-flash' (both work on the new lib)
        self.model_name = "gemini-3-flash-preview"

        self.system_prompt = (
            "You are a witty, sharp-eyed AI assistant with a great sense of humor. "
            "You are my 'eyes' and I am your boss. You must ALWAYS address me as 'Sir'. "
            "You will receive a list of objects detected by a computer vision model. "
            "Your job is to describe the scene accurately but with a funny, friendly, and visual flair. "
            "Don't just list items; paint a picture, crack a mild joke about the combination of objects, "
            "or make a playful observation. Keep it conversational and brief."
        )

    def process_detections(self, detections, user_question="Describe what you see."):
        if not detections:
            return "I don't see any recognizable objects right now."

        # 2. Summarize Data
        object_names = [obj['class_name'] for obj in detections]
        counts = Counter(object_names)
        summary_list = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()]
        scene_description = ", ".join(summary_list)

        # 3. Construct Prompt
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"DETECTED OBJECTS: {scene_description}\n"
            f"USER QUESTION: {user_question}\n"
        )

        try:
            # 4. Generate Content (New Syntax)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            return response.text.strip()

        except Exception as e:
            return f"Error talking to Gemini: {e}"


# --- TEST BLOCK ---
if __name__ == "__main__":
    agent = LLMAgent()

    # Dummy data for testing
    dummy_data = [
        {'class_name': 'person'},
        {'class_name': 'person'},
        {'class_name': 'laptop'}
    ]

    print(f"🧠 Testing new library with model: {agent.model_name}...")
    response = agent.process_detections(dummy_data)
    print(f"\n🤖 Agent Says: {response}")