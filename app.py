#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, send_file,Response

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/emergency')
def emergency_route():
    return send_file('emer.png', mimetype='image/png')

@app.route('/video_feed/<video>')
def video_feed(video):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera(video+".mp4")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True,port=3000)
