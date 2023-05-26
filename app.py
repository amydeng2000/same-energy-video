import os
import cv2
from flask import Flask, render_template, send_from_directory, request, jsonify, redirect
import fnmatch
import os
from random import random
import numpy as np
from PIL import Image
import pandas as pd
import os
import time
import torch
import requests
import base64
import io
import faiss
import pickle
import io
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

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

@app.route('/screenshots/<path:path>')
def send_screenshot(path):
    return send_from_directory('screenshots', path)

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
    with open('data2.pkl', 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)
    index = path_to_index[filepath]
    embedding = np_embeddings[index]
    return embedding

def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename)
                image_files.append((full_filepath, filename))
    return image_files

def embed_image(image_path_or_paths):
    if isinstance(image_path_or_paths, str):
        image_paths = [image_path_or_paths]
    else:
        image_paths = image_path_or_paths

    pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.numpy().tolist()

def write_data():
    np_embeddings = np.array(embeddings).astype('float32')  # Faiss requires float32 data type
    # Create a flat index using the L2 distance metric (Euclidean distance)
    index = faiss.IndexFlatL2(np_embeddings.shape[1])
    # Add the embeddings to the index
    index.add(np_embeddings)

    path_to_index = {}
    index_to_path = {}
    for i, p in enumerate(processed_paths):
        path_to_index[p] = i
        index_to_path[i] = p
        
    # Save the data to disk
    with open('data2.pkl', 'wb') as f:
        pickle.dump((index, np_embeddings, path_to_index, index_to_path), f)

def similarity_search(query_embedding, k=5):
    # Load the saved data from disk
    with open('data2.pkl', 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)

    # Convert the query embedding to float32
    query_embedding = np.array(query_embedding).astype('float32')

    # Perform a similarity search using the faiss index
    _, indices = index.search(np.expand_dims(query_embedding, axis=0), k)
    
    # Retrieve the corresponding image paths for the most similar indices
    similar_paths = [index_to_path[idx] for idx in indices[0]]

    return similar_paths

if __name__ == '__main__':
    video_path = 'videos/example_video.webm' # FIXME
    time_interval = 5 # FIXME

    screenshots_folder = 'screenshots'
    videos_folder = 'videos'
    screenshot_paths = os.listdir(screenshots_folder)
    videos = os.listdir(videos_folder)

    capture_screenshots(video_path, screenshots_folder, time_interval)

    # embed all screenshots
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # pwd = os.getcwd()
    # relative_path = os.path.join(pwd, 'screenshots/')
    paths = find_images_in_folder(screenshots_folder)
    full_paths = [p[0] for p in paths]
    processed_paths = []
    embeddings = []

    i = 0
    for p in range(int(len(full_paths)/10)+1):
        embeddings += embed_image(full_paths[i:i+10])
        processed_paths += full_paths[i:i+10]
        i += 10
        if i % 10 == 0:
            write_data()    
    write_data()

    app.run(debug=True, port=8080)
