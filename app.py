from flask import Flask, render_template, request, redirect, abort, jsonify, make_response
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from stitch import Stitch

input_arr = []

# Convert raw image from client to server without saving images.
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
    global input_arr
    input_arr = []
    return render_template('public/index.html')

@app.route('/display', methods = ['GET','POST'])
def display():
    if (request.files):

        global input_arr
        lst_imgs = request.files.getlist('inputImages')
        content = []
        input_arr = []
        for x in lst_imgs:
            img = Image.open(x.stream)
            img_base64 = img_to_base64_str(img)

            # Convert Pil Image to OpenCV Format for OpenCV Stitching
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            input_arr.append(img)
            content.append(img_base64)
        return render_template('public/display.html', lst_imgs = content)

    else:
        return redirect('/')

@app.route('/stitch', methods = ['GET','POST'])
def stitch():
    if request.method == 'POST':
        try:
            if request.form['checked'] == 'stitch':
                global input_arr
                panorama = Stitch(input_arr)
                result = panorama.fit_transform()
                final_result = panorama.crop(result)

                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                im_pil_result = Image.fromarray(result)
                img_base64_result = img_to_base64_str(im_pil_result)

                final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(final_result)
                img_base64 = img_to_base64_str(im_pil)

            return render_template('public/stitch.html', matched = img_base64, raw = img_base64_result)
        except:
            return redirect('/')
    return redirect('/')

@app.route('/about', methods = ['GET','POST'])
def about():
    return render_template('public/about.html')

if __name__ == '__main__':
    app.run(debug=True)
