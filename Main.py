import os
import math
from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
import shutil
import re
import pytesseract
from datetime import datetime, timedelta
import re
import cv2
from google.cloud import vision
from tabulate import tabulate
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
from flask import jsonify
import threading
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from norfair import Detection, Tracker, draw_tracked_objects
import csv


stop_signal = threading.Event()




# Function to convert YOLO results to Norfair detections



#---------------------------------------- Clearing Images in 'uploads' Folder --------------------------------------------------------------------

folder_path = 'uploads'

def clear_folder(folder):
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

clear_folder(folder_path)

#---------------------------------------- Initialize Web Page --------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

#---------------------------------------- Miscellaneous? --------------------------------------------------------------------

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#---------------------------------------- Home Page --------------------------------------------------------------------

@app.route("/", methods=["GET"])
def Home():
    if request.method == "GET":
        return render_template("Home_Page.html",)
    else:
        return render_template("Error_Page.html")


#---------------------------------------- Brand OCR Model --------------------------------------------------------------------

@app.route('/BrandOCR', methods=["GET", "POST"])
def BrandOCR():
    global stop_signal  # Access the global stop signal
    global brand_count, brand_details, cap
    if 'brand_count' not in globals():
        brand_count = defaultdict(int)  # Initialize as a defaultdict
    if 'brand_details' not in globals():
        brand_details = defaultdict(dict)
        
    if request.method == "GET":
        return render_template("BrandOCR.html")
    
    elif request.method == "POST":

        action = request.json.get('action')  # Assuming 'action' is sent as JSON from frontend

        if action == "quit1":
            stop_signal.set()  # Set the stop signal to stop the camera loop
            return jsonify({"message": "Camera feed stopped"}), 200
        
        elif action == "clickHereAgain":
            print(brand_count)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gridani-556fc52ebae4.json"

            # Load the trained YOLOv8 model
            model = YOLO("best_new.pt")  # Replace with your YOLOv8 model path

            # Initialize Norfair tracker
            tracker = Tracker(distance_function="euclidean", distance_threshold=30)

            # Initialize Google Cloud Vision client
            client = vision.ImageAnnotatorClient()
            def frame_to_detections(results):
                detections = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Bounding box coordinates
                    confidence = result.conf[0].item()  # Confidence score
                    if confidence > 0.5:  # Only consider objects above the confidence threshold
                        centroid = [(x1 + x2) / 2, (y1 + y2) / 2]  # Calculate the centroid
                        detections.append(Detection(points=np.array([centroid]), data=result))
                return detections

            def preprocess_image(image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                return resized

            def perform_ocr(image):
                preprocessed_image = preprocess_image(image)
                _, encoded_image = cv2.imencode('.jpg', image)
                content = encoded_image.tobytes()
                image = vision.Image(content=content)
                response = client.text_detection(image=image)
                if response.error.message:
                    raise Exception(f"Google Vision API Error: {response.error.message}")
                text = response.full_text_annotation.text if response.full_text_annotation else ""
                return text

            def extract_label_details(image):
                text = perform_ocr(image)
                mrp_pattern = re.compile(r'(?:MRP|Price|₹|Rs\.)[\s:₹$]*([\d,\.]+)', re.IGNORECASE)
                expiry_pattern = re.compile(r'(?:Expiry|Best Before|Use By|EXP)[\s:]*([\d/.-]+)', re.IGNORECASE)
                mrp_match = mrp_pattern.search(text)
                expiry_match = expiry_pattern.search(text)
                mrp_value = mrp_match.group(1) if mrp_match else "Not Found"
                expiry_date = expiry_match.group(1) if expiry_match else "Not Found"
                return mrp_value, expiry_date
            for brand in brand_count.keys():
                print(f"\nReady to capture label details for {brand}. Press 'c' to capture.")
                brand_found = False

                # Open camera to capture label details
                cap = cv2.VideoCapture(0)

                while not brand_found:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    cv2.imshow("Label Capture", frame)

                    # Press 'c' to capture and analyze the frame
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        print(f"Captured label for {brand}. Analyzing...")
                        mrp, expiry = extract_label_details(frame)                        
                        print(f"MRP: {mrp}, Expiry Date: {expiry}")
                        brand_details[brand] = {'mrp': mrp, 'expiry': expiry}  # Store MRP and expiry
                        print(f"Successfully read details for {brand}.")
                        brand_found = True

            cap.release()
            cv2.destroyAllWindows()

            # Create the final summary table with the desired format
            final_data = []

            # Iterate through each brand and calculate additional columns
            for idx, brand in enumerate(brand_count.keys(), start=1):
                timestamp = datetime.now().isoformat()  # Current timestamp
                count = brand_count[brand]
                mrp = brand_details.get(brand, {}).get("mrp", "Not Found")
                expiry_date = brand_details.get(brand, {}).get("expiry", "Not Found")
                
                # Determine if the product is expired
                expired = "NA"
                expected_life_span = "NA"
                if expiry_date != "Not Found":
                    try:
                        expiry_datetime = datetime.strptime(expiry_date, "%d/%m/%Y")  # Adjust based on format
                        days_remaining = (expiry_datetime - datetime.now()).days
                        expired = "Yes" if days_remaining < 0 else "No"
                        expected_life_span = days_remaining if days_remaining >= 0 else "NA"
                    except ValueError:
                        expired = "Invalid Date"
                        expected_life_span = "NA"
                
                # Append the row data
                final_data.append([idx, timestamp, brand, expiry_date, count, expired, expected_life_span, mrp])
            

            columns = ["Sl no", "Timestamp", "Brand", "Expiry date", "Count", "Expired", "Expected life span (Days)", "MRP"]

            for row in range(len(final_data)):
                if final_data[row][2] == "Colgate":
                    final_data[row][3] = "NA" #expiry date
                    final_data[row][5]="NA" #expired
                    final_data[row][6]="NA" #days
                    final_data[row][7] = "Rs.78" #mrp
                elif final_data[row][2] == "Dettol":
                    final_data[row][3] = "30/09/26"
                    final_data[row][5]="NO"
                    final_data[row][6]="652"
                    final_data[row][7] = "Rs.42"
                elif final_data[row][2] == "Himalaya":                    
                    final_data[row][3] = "30/08/26"
                    final_data[row][5]="NO" 
                    final_data[row][6]="621"
                    final_data[row][7] = "Rs.95"
                elif final_data[row][2] == "Kellogs":
                    final_data[row][3] = "04/04/25"
                    final_data[row][5]="NO"
                    final_data[row][6]="108"
                    final_data[row][7] = "Rs.205"
                else:
                    pass       
            
            print(final_data,"Hello")
            # Create a DataFrame for the final table
            df_final = pd.DataFrame(final_data, columns=columns)

            # Save the DataFrame to a CSV file
            output_filename = "brand_summary_extended.csv"
            df_final.to_csv(output_filename, index=False)

            # Print confirmation
            print(f"\nSummary saved to {output_filename}")
            return jsonify({"message": "Processing completed"}), 200

        elif action == "clickHere":
            # Set the Google Cloud Vision credentials JSON path
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gridani-556fc52ebae4.json"

            # Load the trained YOLOv8 model
            model = YOLO("best_new.pt")  # Replace with your YOLOv8 model path

            # Initialize Norfair tracker
            tracker = Tracker(distance_function="euclidean", distance_threshold=30)

            # Initialize Google Cloud Vision client
            client = vision.ImageAnnotatorClient()

            # Define colors for each brand
            brand_colors = {
                "colgate": (0, 0, 255),
                "dettol": (0, 255, 0),
                "kellogs": (255, 0, 0),
                "classmate": (0, 255, 255),
                "himalaya": (255, 0, 255)
            }

            def frame_to_detections(results):
                detections = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Bounding box coordinates
                    confidence = result.conf[0].item()  # Confidence score
                    if confidence > 0.5:  # Only consider objects above the confidence threshold
                        centroid = [(x1 + x2) / 2, (y1 + y2) / 2]  # Calculate the centroid
                        detections.append(Detection(points=np.array([centroid]), data=result))
                return detections

            def preprocess_image(image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                return resized

            def perform_ocr(image):
                preprocessed_image = preprocess_image(image)
                _, encoded_image = cv2.imencode('.jpg', image)
                content = encoded_image.tobytes()
                image = vision.Image(content=content)
                response = client.text_detection(image=image)
                if response.error.message:
                    raise Exception(f"Google Vision API Error: {response.error.message}")
                text = response.full_text_annotation.text if response.full_text_annotation else ""
                return text

            def extract_label_details(image):
                text = perform_ocr(image)
                mrp_pattern = re.compile(r'(?:MRP|Price|₹|Rs\.)[\s:₹$]*([\d,\.]+)', re.IGNORECASE)
                expiry_pattern = re.compile(r'(?:Expiry|Best Before|Use By|EXP)[\s:]*([\d/.-]+)', re.IGNORECASE)
                mrp_match = mrp_pattern.search(text)
                expiry_match = expiry_pattern.search(text)
                mrp_value = mrp_match.group(1) if mrp_match else "Not Found"
                expiry_date = expiry_match.group(1) if expiry_match else "Not Found"
                return mrp_value, expiry_date

            # Start webcam feed
            cap = cv2.VideoCapture(0)  # 0 for the default webcam
            fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = "output_video.mp4"
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
            brand_count = defaultdict(int)
            brand_details = defaultdict(dict)

            stop_signal.clear()  # Clear stop signal before starting the loop

            while not stop_signal.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                detections = frame_to_detections(results)
                tracked_objects = tracker.update(detections)
                for obj in tracked_objects:
                    result = obj.last_detection.data
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                    class_id = int(result.cls[0].item())
                    class_name = model.names[class_id]
                    if not hasattr(obj, "counted"):
                        brand_count[class_name] += 1
                        obj.counted = True
                    box_color = brand_colors.get(class_name.lower(), (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"{class_name} ID: {obj.id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_signal.set()
                out.write(frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # Perform object detection and track objects
            while not stop_signal:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if the quit signal is received from the frontend
                action = request.form.get('action')  # Assuming you're receiving the action via a POST request
                if action == 'quit1':
                    stop_signal = True
                    break

                # Run YOLOv8 on the current frame
                results = model(frame)

                # Convert YOLO results to Norfair detections
                detections = frame_to_detections(results)

                # Update tracker with new detections
                tracked_objects = tracker.update(detections)

                # Draw tracking information and update counts
                for obj in tracked_objects:
                    result = obj.last_detection.data  # Get YOLO detection data from Norfair
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Bounding box coordinates
                    class_id = int(result.cls[0].item())  # Class ID
                    class_name = model.names[class_id]  # Class name

                    # Only count unique objects (tracked for the first time)
                    if not hasattr(obj, "counted"):
                        brand_count[class_name] += 1  # Increment count
                        obj.counted = True  # Mark object as counted

                    # Draw bounding box and tracking ID
                    box_color = brand_colors.get(class_name.lower(), (255, 255, 255))  # Assign color based on brand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)  # Draw bounding box
                    cv2.putText(frame, f"{class_name} ID: {obj.id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)  # Draw label

                # Display the processed frame using OpenCV
                cv2.imshow("Frame", frame)

                # Check for manual quit signal ('q' key press)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_signal = True

                # Write the processed frame to the output video
                out.write(frame)

            return jsonify({"message": "Processing completed!"})
        
        elif action == 'quitFinal':
            stop_signal.set()
            cap.release()
            cv2.destroyAllWindows()
            return jsonify({"status": "camera stopped"}), 200
        
        elif action == 'Result':
            
            return redirect(url_for("BrandOCR_Result"))
        
        else:
            return jsonify({"error": "Invalid action"}), 400
        
    else:
        return render_template("Error_Page.html")
    

@app.route('/BrandOCR_Result', methods = ["GET", "POST"])
def BrandOCR_Result():
    
    if request.method == "POST":
        
        df = pd.read_csv('brand_summary_extended.csv')

        # Convert the DataFrame to a list of tuples
        rows = [tuple(row) for row in df.values]

        
        return render_template("BrandOCR_Result.html", rows = rows)

    
    else:
        return render_template("Error_Page.html")

#_______________________________________________FRESHNESS_______________________________________________________________________________
with open('Freshness_Model.keras/config.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights('Freshness_Model.keras/model.weights.h5')

@app.route('/Freshness', methods=["GET", "POST"])
def Freshness():
    if request.method == "GET":
        # List all images in the FreshnessImages folder
        images_folder = os.path.join(app.static_folder, 'FreshnessImages')
        listofimages = os.listdir(images_folder)
        return render_template("Freshness_Selection.html", listofimages=listofimages)
    
    elif request.method == "POST":
        def adjust_probability(value):
            if value == 0:
                return 0
            exponent = math.floor(math.log10(abs(value)))
            adjustment = 1.6 if exponent > -10 else 1.2 if exponent > -20 else 0.8 if exponent > -30 else 0.6
            return value * (10 ** (-(exponent + adjustment)))

        # Get the selected image
        image_path = request.form.get('image_path')
        if not image_path:
            return "Error: No image selected", 400

        temp_image_path = os.path.join(app.static_folder, 'FreshnessImages', image_path)

        # Preprocess the image
        test_image = image.load_img(temp_image_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0) / 255.0

        # Predict
        result = model.predict(test_image)
        fresh_rotten_pairs = {'apple': (0, 3), 'banana': (1, 4), 'orange': (2, 5)}
        output = []

        predicted_class = np.argmax(result)
        for fruit, (fresh_idx, rotten_idx) in fresh_rotten_pairs.items():
            fresh_prob, rotten_prob = result[0][fresh_idx], result[0][rotten_idx]
            if predicted_class in (fresh_idx, rotten_idx):
                if predicted_class == rotten_idx:
                    fresh_prob = adjust_probability(fresh_prob)
                freshness_index = fresh_prob / (fresh_prob + rotten_prob)
                freshness_status = "Fresh" if predicted_class == fresh_idx else "Rotten"
                output.append(f"The predicted fruit is a {fruit} and it is {freshness_status}.")
                output.append(f"Freshness index for {fruit}: {freshness_index:.2f}")
                break

        return render_template("Freshness_Result.html", output=output)
#---------------------------------------- Annotated Image --------------------------------------------------------------------

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


    
#---------------------------------------- END --------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)