# 🔍 The Maker's Lens

**A Real-Time AI Hardware Scanner and Project Architect**

[![Built with FiftyOne](https://img.shields.io/badge/Built%20with-FiftyOne-ff69b4.svg)](https://voxel51.com/fiftyone/)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Gemini%202.5%20Flash-blue.svg)](https://deepmind.google/technologies/gemini/)

## 💡 Inspiration
During hackathons and late-night building sessions, makers often have a desk full of random electronic components—sensors, microcontrollers, resistors, and motors—but struggle to come up with a cohesive project that uses exactly what they have on hand. **The Maker's Lens** turns your webcam into an "Iron Man HUD" that scans your desk, identifies your hardware, and instantly architects a technical project you can build right now.

## ⚙️ What it does
1. **The Eye:** Uses a webcam and OpenCV to capture high-resolution images of electronic components.
2. **The Brain:** Feeds the image to Google's Gemini 1.5 Flash Vision model, which performs zero-shot identification of the hardware and generates a custom, intermediate-to-advanced project blueprint.
3. **The Dashboard:** Automatically logs the captured images, the identified component lists, and the generated project ideas into a beautiful, searchable Voxel51 (FiftyOne) dataset for easy review.

## 🛠️ Tech Stack
* **Python 3.8+**
* **OpenCV** - For real-time webcam video capture.
* **Google Gemini API** - For multimodal vision analysis and natural language generation.
* **Voxel51 (FiftyOne)** - For dynamic dataset visualization and UI generation.
* **python-dotenv** - For secure API key management.

---

## 🚀 Getting Started

Follow these steps to get the project running on your local machine in minutes.

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/makers-lens.git](https://github.com/YOUR_USERNAME/makers-lens.git)
cd makers-lens

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install the required dependencies
pip install fiftyone opencv-python google-generativeai pillow python-dotenv
```

### 2. Secure Your API Key
Get a free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/). 

Create a file named `.env` in the root of your project directory and add your key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
*(Note: The `.env` file is included in our `.gitignore` to keep your credentials safe!)*

### 3. Run the Application
Make sure your webcam is connected, then start the scanner:
```bash
python app.py
```

---

## 🎮 How to Demo

1. Once the app is running, a webcam window will pop up. Point your camera at a breadboard, an Arduino, or a pile of loose sensors.
2. Press the **Spacebar** to take a snapshot. Look at your terminal to confirm the AI is analyzing it.
3. Press **`q`** to close the camera interface.
4. The **FiftyOne Dashboard** will instantly launch in your web browser (`http://localhost:5151`). Click on the images you just captured to see the AI's component breakdown and project blueprint side-by-side with your photo!

## 🚧 Challenges We Ran Into
* **Real-time constraints:** Training a custom YOLO or CNN model to recognize specific microcontrollers requires thousands of labeled images and massive compute. We pivoted to using Gemini 1.5 Flash to achieve zero-shot object detection within the 2-hour hackathon limit.
* **UI Development:** Building a React/Vue frontend from scratch would have eaten our entire time budget. Voxel51 provided an out-of-the-box, highly professional UI that allowed us to focus purely on the computer vision and AI logic.

## 🔮 What's Next
* **Voice Integration:** Allowing the user to ask questions about the scanned components (e.g., "What is the pinout for that specific IC?") via audio.
* **Live AR Overlays:** Rendering bounding boxes and labels directly over the webcam feed before capturing the image.
