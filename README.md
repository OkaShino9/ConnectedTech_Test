# Blue Water Bottle Detection & Counting

This repository contains the solution for the AI Engineer selection test. The objective is to accurately detect and count the number of blue water bottles inside a plastic bag.

## Features
- **YOLOv8 Object Detection**: Utilizes a custom-trained YOLOv8 model for accurate detection of blue water bottles.
- **Streamlit Web Application**: Provides a simple and interactive UI to upload or select test images, run the model, and display the detection results and count.

## Installation & Setup

1. **Clone or Extract the Source Code**
   Ensure all files (including `best11s.pt`, images, `app.py`, and `requirements.txt`) are in the same directory.

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```
   This will start a local web server. Open the URL provided in the terminal (usually `http://localhost:8501`) to interact with the application.

## How to Reproduce Results
1. Launch the Streamlit app.
2. In the left sidebar under "Simulation Options", select an image (`Data_Bottles.png` or `Example_Test_Bottles.png`).
3. Click on the **"Detect and Count Bottles"** button.
4. The system will process the image with the YOLOv8 model and display the bounding boxes along with the total count of the detected blue water bottles.

---

## Experimental Results and Observations

During the development of this solution, several approaches were tested and evaluated to achieve the best performance:

1. **Initial Approach (Preprocessing over Postprocessing)**: 
   My initial focus was heavily invested in image preprocessing techniques (e.g., color thresholding, exposure adjustment) to isolate the blue bottles from the plastic bag glare. However, I realized I had overlooked the post-processing aspect. I later found out that simply feeding the raw image directly to the trained detection model was sufficient and yielded excellent results. This initial misdirection cost me a significant amount of time.

2. **Pure OpenCV Detection**:
   I attempted to use purely traditional computer vision methods via OpenCV (color masking, morphological operations, and contour detection). Unfortunately, this method proved to be highly unreliable, resulting in numerous false negatives. The varying lighting conditions, reflections, and occlusions caused by the plastic bag made generic rule-based computer vision struggle.

3. **SAHI (Slicing Aided Hyper Inference)**:
   I also experimented with SAHI to better handle occlusions and potentially detect smaller or partially hidden features. However, it did not provide a substantial improvement for this specific use case and ended up complicating the pipeline without a corresponding performance gain.

4. **Vision Language Models (VLMs) with ResNet Backbone**:
   Interestingly, I explored using Vision Language Models (VLMs) combined with a ResNet backbone. In my tests, this approach yielded the most promising results, achieving zero false negatives and perfectly detecting the blue water bottles inside the bag.

### Conclusion

This challenge was an excellent learning experience. Although I had to work on this assignment during my midterm exam week—which limited the time I could dedicate to experimenting with every alternative approach—I gave it my absolute best effort to implement the most effective solutions I could think of.

I sincerely hope to have the opportunity to join the team and work alongside the senior engineers at the company. I am highly driven to develop my skills further, learn from experienced professionals, and discover the best engineering practices for solving complex computer vision problems like this one.
