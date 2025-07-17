# Player Re-Identification in Sports Footage 

### **Project Overview**

This project implements a solution for **Task Option 2: Re-Identification in a Single Feed**. The goal is to detect and track players in a 15-second video clip (`15sec_input_720p.mp4`), ensuring that players who become occluded (go out of frame) and later reappear are assigned their original ID.

The solution uses the provided YOLOv11 model for player detection and a custom tracker that combines a Kalman Filter for motion prediction with a deep learning-based appearance feature extractor for re-identification.

---

### **Setup and Installation**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd Liat.ai-AI_Intern_Assignment_YourName
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Assets:**
    * Place the input video `15sec_input_720p.mp4` in the main project directory.
    * Download the object detection model from the provided link.
    * Create a `models/` directory and place the downloaded `yolov11_players_ball.pt` file inside it.

---

### **How to Run the Code**

Execute the main script from the root directory of the project:

```bash
python src/main.py --video_path 15sec_input_720p.mp4 --model_path models/yolov11_players_ball.pt --output_path output/result.mp4
```

The processed video with player IDs and bounding boxes will be saved to `output/result.mp4`.
