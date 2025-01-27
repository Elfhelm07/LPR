import requests
import json

url = "http://localhost:5000/process_video"
data = {
    "video_path": "E:\\QRLPR\\final (1).mp4"  # Replace with your actual video path
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())