# 👁️ Eye of LLM
An AI Assistant with Vision Capabilities. Powered by YOLO11 (Vision), Gemini 2.5 Flash (Reasoning), and Python.

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






Tool call (from LLM to SAM):
```json
{
  "tool": "sam.segment",
  "args": {
    "image_url": "https://...",
    "prompt_type": "box",        
    "prompt": ["x1","y1","x2","y2"] 
  }
}
```

SAM response (to LLM):
```json
{
  "masks": ["base64_png_1"],
  "polygons": [["x","y"],"..."],
  "bboxes": [["x1","y1","x2","y2"], "..."],
  "scores": [0.98, "..."]
}
```

---

## Example code (key files)

### `requirements.txt`

```
requests
python-dotenv
Pillow
numpy
opencv-python
google-generativeai>=0.2.0
```

### `.env.example`

```
HF_API_KEY=hf_...
GEMINI_API_KEY=...
HF_SAM_MODEL=facebook/sam-vit-huge
GEMINI_MODEL=gemini-1.5-pro
```