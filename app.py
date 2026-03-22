import cv2
import fiftyone as fo
from google import genai
import os
import time
import PIL.Image
from dotenv import load_dotenv

os.environ["QT_QPA_PLATFORM"] = "xcb" 
os.environ["QT_STYLE_OVERRIDE"] = ""           # Stops the 'kvantum' warning
os.environ["QT_LOGGING_RULES"] = "*=false"     # Mutes the font warnings
# --- 1. SETUP API ---
# Load the API key securely from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found! Make sure .env is in your Voxol51_hackthon folder.")

# Initialize the NEW GenAI client
client = genai.Client(api_key=api_key)
model_id = 'gemini-2.5-flash'

# --- 2. SETUP FIFTYONE DATASET ---
dataset_name = "makers-lens-scans"
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

def capture_and_analyze():
    print("📸 Starting The Maker's Lens...")
    print("👉 Press 'Spacebar' to scan components.")
    print("👉 Press 'q' to quit the camera and launch the dashboard.")
    
    cap = cv2.VideoCapture(0) 
    
    # Allow the camera a moment to warm up
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from webcam. Is another app using it?")
            break
            
        cv2.imshow("The Maker's Lens - Live Scanner", frame)
        
        key = cv2.waitKey(1)
        if key == 32: # Spacebar pressed
            image_filename = f"scan_{int(time.time())}.jpg"
            image_path = os.path.abspath(image_filename)
            
            cv2.imwrite(image_path, frame)
            print(f"\n🔍 Image captured! Analyzing components...")
            
            # Send to AI using the new SDK
            try:
                img = PIL.Image.open(image_path)
                prompt = """
                1. List the electronic components you see in this image.
                2. Suggest one intermediate-to-advanced project idea using only these components. 
                Focus on technical implementation details. Consider architectures involving Real-Time 
                Operating Systems, embedded C/C++, serial protocols (I2C/SPI/UART), or control loops, 
                if the hardware supports it. Keep it brief and highly technical.
                """
                
                # New Generate Content Syntax
                response = client.models.generate_content(
                    model=model_id,
                    contents=[prompt, img]
                )
                ai_text = response.text
                print("✅ AI Analysis Complete!")
                
            except Exception as e:
                ai_text = f"API Error: {e}"
                print(f"❌ Error communicating with AI: {e}")
            
            # Add to FiftyOne Dashboard
            sample = fo.Sample(filepath=image_path)
            sample["ai_analysis"] = ai_text
            dataset.add_sample(sample)
            print("💾 Saved to FiftyOne dataset.")
            
        elif key == ord('q'):
            print("\n🛑 Closing camera...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_analyze()
    print("🚀 Launching FiftyOne Dashboard...")
    print("🌐 Open http://localhost:5151 in your browser if it doesn't open automatically.")
    session = fo.launch_app(dataset)
    session.wait()
