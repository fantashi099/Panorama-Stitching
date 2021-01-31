from flask import Flask, render_template, request, redirect, abort, jsonify, make_response
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def main():
    return render_template('public/index.html')

@app.route('/display', methods = ['POST'])
def display():
    if (request.files):

        lst_imgs = request.files.getlist('inputImages')
        content = []
        for x in lst_imgs:
            img = Image.open(x.stream)
            img_base64 = img_to_base64_str(img)
            content.append(img_base64)
        return render_template('public/display.html', lst_imgs = content)


if __name__ == '__main__':
    app.run(debug=True)
