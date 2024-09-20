"""
rest API for materials object detection in CV.KHS
"""

import io
import os
import cv2
import json
import time
import uuid
import argparse
import pandas as pd
import threading
from PIL import Image
from datetime import datetime, date
from os.path import dirname, join, abspath
from src.prediction import detect, get_img_size
from werkzeug.exceptions import HTTPException
from flask import Flask, request, session, json
from src.to_dashboard.main import update_api_status, append_data, save_img, connect_db, saveImgReal

lock = threading.Lock()


app = Flask(__name__)
app.secret_key = 'yolov5rev2'
app.config["DEBUG"] = True


path = os.getcwd()
path_temp = os.path.join(path , "temp")

DETECTION_URL = "/v1/materials_detection"


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    response = e.get_response()
    response.data = json.dumps({
        "success": False,
        "code": e.code,
        "name": e.name,
        "message": e.description,
        "data": []
    })
    response.content_type = "application/json"
    return response

@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    base_path = dirname(abspath(__file__))
    weight_path = join(base_path, 'weight_type')
    temp_path = join(base_path, 'temp')
    bankimg_path = '/usr/share/nginx/html/object-countings/bank-img'
    saveImgReal_path = '/usr/share/nginx/html/object-countings/real-bank-img'
    temp_file = join(temp_path, f'{str(uuid.uuid1())}-img_process.jpg')
    temp_json = join(temp_path, f'{str(uuid.uuid1())}-result.json')
    start_time = time.time()
    
    if request.method != "POST":
        return "QUICK MATERIALS DETECTION"
    image_file = request.files.getlist("image")
    for img in image_file:
        image = img.read()
        with open(temp_file, 'wb')  as outfile:
            outfile.write(image)
        if request.form.get('obj_type'):
            obj_type = request.form.get('obj_type')
            if os.path.exists(join(weight_path, obj_type+ '.pt')):
                weight_path = join(weight_path, obj_type+ '.pt')
        
        result = detect(source=temp_file, weights=weight_path)
        result.to_json(temp_json, orient='records')
        f = open(temp_json)
        data_json = json.load(f)
        
        im_width, im_height = get_img_size(temp_file)
        data = {
            "success": True,
            "message": "Successfully",
            "img_width": im_width,
            "img_height": im_height,
            "code": 200
        } 
        
        ip_addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        append_data(
            table_name = 'analytics_services_api',
            model = obj_type,
            id_api = 3,
            ip_address = ip_addr,
            request_date = datetime.now(),
            url_api = "http://localhost:2060/v1/materials_detection",
            response = data,
            response_time = round((time.time() - start_time) * 100 )
        )
        data['data'] = data_json
        last_id = append_data(
            table_name = 'analytics_counting_object',
            model = obj_type,
            ip_address = ip_addr,
            request_date = datetime.now(),
            pred_transact = len(data_json),
            response_time = round((time.time() - start_time) * 100 ),
            img_path = 'None'
        )
        data['id_counting'] = last_id
        data['time'] = str(round(time.time() - start_time, 2))+'s'
        os.remove(temp_file)
        os.remove(temp_json)
        return data

def detect_fetch(temp_file, weight_path):
    result = detect(source=temp_file, weights=weight_path)
    return result

        
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
        parser.add_argument("--port", default=2060, type=int, help="port number")
        args = parser.parse_args()
        update_api_status(3, 'Active')
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    finally :
        update_api_status(3, '')