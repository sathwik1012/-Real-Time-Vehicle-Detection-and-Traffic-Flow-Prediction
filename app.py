from flask import Flask, request, render_template, redirect, url_for, session 
import cv2 
import numpy as np 
import time 
from ultralytics import YOLO 
import os 
app = Flask(__name__) 
app.secret_key = 'your_secret_key_here' 
model = YOLO("yolov8x.pt") 
vehicle_classes = [ 
"car", "motorcycle", "motorbike", "truck", "bus", "auto", "bicycle",  
"ambulance", "fire truck", "police car", "emergency vehicle" 
] 
priority_vehicles = { 
"ambulance": 5, 
"fire truck": 5, 
"police car": 4, 
"emergency vehicle": 5 
} 
vehicle_name_mapping = { 
"emergency vehicle": "ambulance", 
"fire engine": "fire truck", 
"police vehicle": "police car" 
} 
def preprocess_image(image_path): 
    image = cv2.imread(image_path) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_resized = cv2.resize(image_rgb, (1280, 1280)) 
    return image_resized 
def calculate_time(vehicle_counts): 
    time_per_vehicle = { 
    "car": 1, 
    "motorcycle": 1, 
    "motorbike": 1, 
    "truck": 2, 
    "bus": 2, 
    "auto": 1, 
    "bicycle": 1, 
    "ambulance": 5, 
    "fire truck": 5,
    "police car": 4, 
    "emergency vehicle": 5 
    }  
    total_time = 0 
    for vehicle, count in vehicle_counts.items(): 
        if vehicle in priority_vehicles: 
            total_time += time_per_vehicle.get(vehicle, 0) * count * 2 
        else: 
            total_time += time_per_vehicle.get(vehicle, 0) * count 
     
    return total_time 
 
def detect_vehicles(image): 
    start_time = time.time() 
     
    results = model(image, imgsz=1280, conf=0.4, iou=0.5) 
    detection_time = time.time() - start_time 
     
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes} 
     
    for box, conf, cls in zip(results[0].boxes.xywh, results[0].boxes.conf, results[0].boxes.cls): 
        class_name = results[0].names[int(cls)].lower() 
         
        if class_name in vehicle_name_mapping: 
            class_name = vehicle_name_mapping[class_name] 
         
        if class_name == "motorcycle": 
            class_name = "motorbike" 
         
        if class_name in vehicle_counts: 
            if class_name in priority_vehicles: 
                if conf > 0.5: 
                    vehicle_counts[class_name] += 1 
            else: 
                vehicle_counts[class_name] += 1 
     
    vehicle_counts = {k: v for k, v in vehicle_counts.items() if v > 0} 
     
    estimated_time = calculate_time(vehicle_counts) 
     
    return vehicle_counts, results[0].plot(), detection_time, estimated_time 
 
@app.route("/", methods=["GET", "POST"]) 
def upload_files(): 
    if request.method == "POST": 
        uploaded_files = []
        for i in range(1, 5): 
            file = request.files.get(f"file{i}") 
            if file: 
                os.makedirs("uploads", exist_ok=True) 
                os.makedirs(os.path.join("static", "results"), exist_ok=True)     
                file_path = os.path.join("uploads", f"road_{i}.jpg") 
                file.save(file_path) 
                uploaded_files.append(file_path) 
        results = [] 
        for i, file_path in enumerate(uploaded_files, start=1): 
            image = preprocess_image(file_path) 
            vehicle_counts, result_image, detection_time, estimated_time = detect_vehicles(image) 
 
            total_count = sum(vehicle_counts.values()) 
            result_path = os.path.join("static", "results", f"result_{i}.jpg") 
            cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)) 
 
            results.append({ 
                "vehicle_counts": vehicle_counts, 
                "result_image": result_path, 
                "detection_time": round(detection_time, 2), 
                "total_count": total_count, 
                "estimated_time": estimated_time 
            }) 
             
            os.remove(file_path) 
 
        results = sorted(results, key=lambda x: x['total_count'], reverse=True) 
         
        session['detection_results'] = results 
         
        return render_template("result.html", results=results) 
 
    return render_template("upload.html") 
 
@app.route("/output") 
def output_page(): 
    results = session.get('detection_results', []) 
    return render_template("output.html", results=results) 
 
if __name__ == "__main__": 
    app.run(debug=True) 