import flask 
from flask import Flask, render_template, request
#from prediction import detect_and_classification
from inference import ObjectDetection
import os
import base64
from io import BytesIO
import cv2
from PIL import Image
            
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
   return render_template('start.html')

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    errors = []
    allowed_extensions = set(['png', 'jpg', 'jpeg', 'gif'])
    
    try:
        if request.method == 'POST':
            currentfile = request.files.get('file', '')
    except:
        errors.append(
                    "Unable to read file. Please make sure it's valid and try again."
                    )
    # prediction of model
    class_name = ObjectDetection()
    image_result,type = class_name.__call__(currentfile) 

    #image_result, quantity, quality = detect_and_classification(currentfile, threshold = .5)   

    buffered = BytesIO()
    image_result = Image.fromarray(image_result)
    image_result = image_result.resize((400, 400))
    image_result.save(buffered, format="JPEG")
    image_memory = base64.b64encode(buffered.getvalue())

    #return render_template("result.html", quantity=quantity, quality=quality, img_data=image_memory.decode('utf-8'))
    return render_template("result.html", type=type, img_data=image_memory.decode('utf-8'))



if __name__ == '__main__':
    
    app.run(debug=True)
