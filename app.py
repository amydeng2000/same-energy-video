import cv2
from flask import Flask, render_template, send_from_directory, request, jsonify
import os
from random import random
import numpy as np
import pandas as pd
import pickle
from google.cloud import storage

app = Flask(__name__)

def get_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return frame_rate

def capture_screenshots(video_path, output_folder, time_interval):
    # calculate frame interval
    fps = get_frame_rate(video_path)
    interval = time_interval * fps
    video = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    screenshot_count = 0

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % interval == 0:
            screenshot_path = os.path.join(output_folder, "screenshot_{}.jpg".format(screenshot_count))
            cv2.imwrite(screenshot_path, frame)
            screenshot_count += 1

    video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    # screenshots_folder = 'screenshots' # FIXME
    # screenshots = os.listdir(screenshots_folder)
    return render_template('index.html', screenshot_paths=screenshot_paths, videos=videos)

# @app.route('/screenshots/<path:path>')
# def send_screenshot(path):
#     return send_from_directory('screenshots', path)

def generate_public_url(bucket_name, blob_name):
    """Generate a public URL for a blob. """
    url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    return url

@app.route('/screenshots/')
def send_screenshot():
    # bucket_name = "<Your Google Cloud Storage Bucket Name>"
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, max_results=10) # Start with 20 blobs

    screenshot_urls = [generate_public_url(bucket_name, blob.name) for blob in blobs]
    next_token = blobs.next_page_token if blobs.next_page_token else None
    return jsonify({"screenshot_urls": screenshot_urls, "next_token": next_token})

@app.route('/screenshots/next/<string:token>')
def get_next_screenshot(token):
    # bucket_name = "<Your Google Cloud Storage Bucket Name>"
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, max_results=10, page_token=token) # Use the token to get the next page

    screenshot_urls = [generate_public_url(bucket_name, blob.name) for blob in blobs]
    next_token = blobs.next_page_token if blobs.next_page_token else None
    return jsonify({"screenshot_urls": screenshot_urls, "next_token": next_token})

@app.route('/videos')
def send_video():
    video_path = 'videos/example_video.webm'
    return send_from_directory('', video_path)

@app.route('/resort_screenshots', methods=['POST'])
def resort_screenshots():
    screenshot_path = request.json['path']
    screenshot_path = screenshot_path.strip("/")   # FIXME
    # absolute_path = relative_path + screenshot_path
    screenshot_embedding = get_embedding(screenshot_path)
    new_screenshot_paths = similarity_search(screenshot_embedding, len(screenshot_paths))
    return jsonify(screenshot_paths=new_screenshot_paths)

# Retrieve the embedding for a given screenshot_path
def get_embedding(filepath):
    with open('data.pkl', 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)
    index = path_to_index[filepath]
    embedding = np_embeddings[index]
    return embedding

def similarity_search(query_embedding, k=5):
    # Load the saved data from disk
    with open('data.pkl', 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)

    # Convert the query embedding to float32
    query_embedding = np.array(query_embedding).astype('float32')

    # Perform a similarity search using the faiss index
    _, indices = index.search(np.expand_dims(query_embedding, axis=0), k)
    
    # Retrieve the corresponding image paths for the most similar indices
    similar_paths = [index_to_path[idx] for idx in indices[0]]

    return similar_paths

if __name__ == '__main__':
    bucket_name = "same-energy-screenshots-storage"

    video_path = 'videos/example_video.webm' # FIXME
    time_interval = 5 # FIXME

    screenshots_folder = 'screenshots'
    videos_folder = 'videos'
    screenshot_paths = os.listdir(screenshots_folder)
    videos = os.listdir(videos_folder)

    capture_screenshots(video_path, screenshots_folder, time_interval)

    app.run(debug=True, port=8080)
