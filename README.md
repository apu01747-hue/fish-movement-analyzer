# Fish Movement Analyzer v1.0

A Streamlit web app based on the current Colab workflow:
- Upload an MP4/AVI/MOV video.
- Select the centre of the green, red and orange marker in a reference frame.
- Track each marker forward and backward using HSV colour segmentation, contour filtering and nearest-candidate matching.
- Interpolate missing positions only between detected points.
- Calculate estimated path distance and speed in pixels and pixels/second.
- Export trajectory/speed plots, tracked MP4, CSV data and a Word report.

## Run locally

1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open a terminal in the extracted folder.
4. Run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit, usually http://localhost:8501.

## Deploy as a website

For Streamlit Community Cloud:
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open https://share.streamlit.io/ and deploy the repository.
4. Select `app.py` as the main file.

For larger videos or heavier processing, a server with more memory/CPU may be needed. Public hosting may have upload and execution limits.

## Important scientific limitations

- This version tracks the added coloured circles, not the fish body independently.
- The marker must remain aligned with its fish throughout the clip.
- Nearest-candidate matching can switch to another similar-coloured object or lose the marker.
- Interpolated points are estimates, not actual observed positions. Long gaps should be reviewed.
- Results are in pixels and pixels/second. Physical units require a known scale in the video.
- Review the tracked video before using the measurements in research.

## Current detection settings

HSV thresholds and contour filters are starting values. They may need tuning for different videos, lighting, marker colours and video compression.
