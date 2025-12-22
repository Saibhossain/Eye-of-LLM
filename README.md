# 👁️ Eye of LLM
An AI Assistant with Vision Capabilities. Powered by YOLO11 (Vision), Gemini 2.5 Flash (Reasoning), and Python.

## How the full system works
           You ask a question
                   ↓
           Camera takes photo
                   ↓
      YOLO detects objects (hand,phone,etc)
                   ↓
         Objects converted to text:
            - "2 persons"
            - "1 laptop"
                   ↓
         Gemini receives:
            - detected objects
            - your question
                   ↓
        Gemini answers USING VISION



## 📖 Overview
Eye of LLM bridges the gap between Computer Vision and Large Language Models. It allows an AI agent to "see" the real world through your webcam.

1. The Eyes: Uses YOLO11 (Ultralytics) to detect objects in real-time with high speed and accuracy (optimized for Mac M1/M2).

2. The Brain: Sends the detected object data to Google Gemini 2.5 Flash, which reasons about the scene.

3. The Personality: The AI replies with a witty, helpful personality (configurable), addressing the user as "Sir" and making observations about the environment.

---

## 🚀 Features
1.Real-Time Detection: Instantly identifies objects (People, Laptops, Phones, Cups, etc.) at 30+ FPS.

2. Visual Interface: Draws bounding boxes and labels directly on the video feed.

3. Interactive AI: Press SPACE to freeze the moment and ask the AI what it thinks.

4. Contextual Awareness: The AI doesn't just list objects; it understands the context ("A person with a suitcase" → "Looks like you are travelling").

5. Mac M1/M2 Optimized: Runs efficiently using Apple Silicon hardware acceleration (MPS/CPU).

---

## 📂 Project Structure

```
llm-with-eye/
├── app/
│   ├── __init__.py
│   ├── camera.py        # Handles Webcam I/O and threading
│   ├── main.py          # Entry point to run the application
│   ├── sam_agent.py     # Vision Agent (Runs YOLO11 detection logic)
│   ├── llm_agent.py     # Brain Agent (Connects to Google Gemini API)
│   └── controller.py    # Main Logic Loop (Connects Vision <-> LLM <-> UI)
├── .env                 # API Keys (Not uploaded to GitHub)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🛠️ Installation

1. Clone the Repository
```bash
git clone https://github.com/Saibhossain/Eye-of-LLM.git
cd Eye-of-LLM
```
2. Set up Virtual Environment
It is recommended to use a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Configure API Keys
Create a **.env** file in the root directory:
```bash
touch .env
```
* Open it and add your Google Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```
(Get a free key from Google AI Studio) link-(https://aistudio.google.com/usage?timeRange=last-28-days)


## 💻 Usage
Run the Application
```bash
python main.py
```
Controls

🟢 Live View: The camera will open and start detecting objects immediately.

⌨️ SPACEBAR: Press to trigger the "Brain." The AI will analyze the current objects and print a response on the screen.

🔴 Q: Press q to quit the application.




## 📊 Data Flow & JSON Structure
Understanding how the "Eyes" talk to the "Brain" is key to customizing this project.
1. Visual Agent Output
```json
[
  {
    "object": "person",
    "accuracy": 0.88,
    "bbox": [100, 50, 200, 300]
  },
  {
    "object": "laptop",
    "accuracy": 0.92,
    "bbox": [150, 300, 400, 500]
  },
  {
    "object": "coffee cup",
    "accuracy": 0.75,
    "bbox": [410, 320, 450, 380]
  }
]
```
2. LLM Processing

The LLMAgent receives this list and summarizes it into natural language context for the AI prompt:

Raw Input: 
```json
[{'object': 'person'}, {'object': 'person'}, {'object': 'cup'}]
```
Summarized Context: "2 persons, 1 cup"

Final Prompt to Gemini:

"RAW DATA: 2 persons, 1 cup. MY QUESTION: What do you see?"

---

## Example code (key files)

### `requirements.txt`

```
requests==2.32.5
python-dotenv==1.2.1
Pillow==12.0.0
numpy==2.2.6
opencv-python==4.12.0.88
matplotlib==3.10.8
ultralytics==8.3.241

```

### `.env`

```
GEMINI_API_KEY=...
```