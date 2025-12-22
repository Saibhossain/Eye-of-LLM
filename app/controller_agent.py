from camera import capture_image
from sam_agent import detect_objects
from llm_agent import LLMAgent

class VisionLLMAgent:
    def __init__(self):
        self.llm = LLMAgent()

    def run(self, user_question):
        # 1. SEE
        frame = capture_image()

        # 2. DETECT OBJECTS
        detections = detect_objects(frame)

        # 3. THINK & ANSWER
        answer = self.llm.process_detections(
            detections=detections,
            user_question=user_question
        )

        return answer


# Test controller alone
if __name__ == "__main__":
    agent = VisionLLMAgent()
    print(agent.run("What do you see?"))
