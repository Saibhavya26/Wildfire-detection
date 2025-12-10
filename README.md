# AI-Based Wildfire Detection System

This project implements a real-time wildfire detection and spread prediction system using Deep Learning and Computer Vision.

## Technologies Used
- Python  
- OpenCV  
- NumPy  
- YOLO (for fire detection)  
- Optical Flow (for fire movement tracking)

## Project Description
This system detects wildfire from live video streams using a trained YOLO model.  
Optical Flow is used to estimate the direction and movement of fire.  
The detected fire region is mapped onto a grid to simulate spread direction and intensity.

## Features
- Real-time fire detection
- Fire direction tracking using optical flow
- Fire spread simulation using grid-based modeling
- Supports disaster response and early warning systems

## How to Run
```bash
pip install -r requirements.txt
python main.py
