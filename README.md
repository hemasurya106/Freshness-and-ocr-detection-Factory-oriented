**Brand OCR and Freshness Detection System**

**Overview**

This project is a web application that utilizes computer vision techniques to perform Optical Character Recognition (OCR) on product labels and assess the freshness of fruits. The application leverages YOLOv8 for object detection, Google Cloud Vision for OCR, and a custom TensorFlow model for freshness classification. The results are displayed in a user-friendly web interface built with Flask.

**Features**

Brand OCR: Capture and analyze product labels to extract details such as MRP and expiry dates.
Freshness Detection: Classify fruits as fresh or rotten based on images uploaded by the user.
Real-time Object Tracking: Use Norfair to track detected objects in real-time.
Data Summary: Generate a summary of detected brands and their details, which can be exported as a CSV file.
Web Interface: A simple and intuitive web interface for users to interact with the application.

**Technologies Used**

Backend: Flask
Computer Vision: OpenCV, YOLOv8, Google Cloud Vision API
Machine Learning: TensorFlow, Keras
Data Handling: Pandas, NumPy
Frontend: HTML, CSS, JavaScript
Database: CSV files for storing results
Installation

**Prerequisites**

Python 3.7 or higher
pip (Python package installer)
Google Cloud account with Vision API enabled

**Usage**

**Brand OCR:**

Navigate to the Brand OCR page.
Click on "Click Here" to start the camera feed.
Capture product labels by pressing 'c' and analyze the details.
The results will be displayed and can be exported as a CSV file.

**Freshness Detection:**

Navigate to the Freshness page.
Select an image of a fruit from the provided list.
The application will classify the fruit as fresh or rotten and display the results.
Acknowledgments

**YOLOv8 for object detection.**
Google Cloud Vision API for OCR capabilities.
Norfair for object tracking.
Flask for the web framework.
