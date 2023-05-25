import os
import cv2
from flask import Flask, render_template, send_from_directory, request
import subprocess

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
    return render_template('index.html', screenshots=screenshots, videos=videos)

@app.route('/screenshots/<path:path>')
def send_screenshot(path):
    return send_from_directory('screenshots', path)

@app.route('/videos')
def send_video():
    video_path = 'videos/example_video.webm'
    return send_from_directory('', video_path)


# @app.route('/play_video', methods=['POST'])
# def play_video():
#     screenshot_path = request.form['screenshot_path']
#     video_path = 'example_video.webm' # FIXME
#     screenshot_time = int(''.join(filter(str.isdigit, screenshot_path)))
#     print("*************************************")
#     print(screenshot_time)
#     print("*************************************")
#     cmd = ['ffplay', '-ss', str(screenshot_time), '-i', video_path, '-ss', str(screenshot_time)]
#     subprocess.Popen(cmd)
#     return 'Success'

if __name__ == '__main__':
    video_path = 'videos/example_video.webm' # FIXME
    time_interval = 5 # FIXME

    screenshots_folder = 'screenshots'
    videos_folder = 'videos'
    screenshots = os.listdir(screenshots_folder)
    videos = os.listdir(videos_folder)
    app.logger.info(videos)

    capture_screenshots(video_path, screenshots_folder, time_interval)

    app.run(debug=True, port=8080)
