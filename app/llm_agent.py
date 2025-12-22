import os
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter


load_dotenv()


class LLMAgent:
    def __init__(self):
        # 1. Setup Gemini
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ API Key missing! Check your .env file.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # System instruction to give the LLM a personality
        self.system_prompt = (
            "You are a helpful AI assistant with vision capabilities. "
            "You cannot see the image directly, but you receive a list of objects "
            "detected by a YOLO computer vision model. "
            "Describe what is happening based on these objects. Be concise and natural."
        )

    def process_detections(self, detections, user_question="Describe what you see."):
        """
        Takes the raw YOLO output list and sends a summary to Gemini.
        """
        if not detections:
            return "I don't see any recognizable objects right now."

        # 2. Summarize the data (Turn JSON into English)
        # Extract just the class names (e.g., ['person', 'person', 'bowl'])
        object_names = [obj['class_name'] for obj in detections]

        # Count them (e.g., {'person': 2, 'bowl': 1})
        counts = Counter(object_names)

        # Create a readable string (e.g., "2 persons, 1 bowl")
        summary_list = []
        for name, count in counts.items():
            s = "s" if count > 1 else ""  # Handle plural
            summary_list.append(f"{count} {name}{s}")

        scene_description = ", ".join(summary_list)

        # 3. Construct the Prompt
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"DETECTED OBJECTS: {scene_description}\n"
            f"USER QUESTION: {user_question}\n"
        )

        try:
            # 4. Ask Gemini
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error talking to Gemini: {e}"


# --- TEST BLOCK (Run this file to test independently) ---
if __name__ == "__main__":
    # Simulate the input you provided
    dummy_data = [
        {'bbox': [745, 153, 1274, 711], 'confidence': 0.91, 'class_name': 'person'},
        {'bbox': [154, 236, 951, 717], 'confidence': 0.84, 'class_name': 'person'},
        {'bbox': [0, 683, 52, 719], 'confidence': 0.30, 'class_name': 'bowl'}
    ]

    agent = LLMAgent()

    print("🧠 Thinking...")
    response = agent.process_detections(dummy_data)
    print(f"\n🤖 Agent Says: {response}")