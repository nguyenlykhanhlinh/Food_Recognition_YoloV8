import os
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

model = YOLO("C:/Users/MyPC/Documents/PYTHON/Data_Foo/train2/weights/best.pt")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

class_names = ['rice','eels on rice','pilaf',"chicken-'n'-egg on rice",'pork cutlet on rice','beef curry','sushi',
               'chicken rice','fried rice','tempura bowl','bibimbap','toast','croissant','roll bread','raisin bread','chip butty',
               'hamburger','pizza','sandwiches','udon noodle','tempura udon','soba noodle','ramen noodle','beef noodle',
            'tensin noodle','fried noodle','spaghetti','Japanese-style pancake','takoyaki','gratin','sauteed vegetables',
               'croquette','grilled eggplant','sauteed spinach','vegetable tempura','miso soup','potage','sausage',
               'oden','omelet','ganmodoki','jiaozi','stew','teriyaki grilled fish','fried fish','grilled salmon',
             'salmon meuniere ','sashimi','grilled pacific saury','sukiyaki','sweet and sour pork','lightly roasted fish',
               'steamed egg hotchpotch','tempura','fried chicken','sirloin cutlet','nanbanzuke','boiled fish','seasoned beef with potatoes',
               'hambarg steak','beef steak','dried fish','ginger pork saute',
             'spicy chili-flavored tofu','yakitori','cabbage roll','rolled omelet','egg sunny-side up','fermented soybeans',
               'cold tofu','egg roll','chilled noodle','stir-fried beef and peppers','simmered pork','boiled chicken and vegetables',
               'sashimi bowl','sushi bowl','fish-shaped pancake with bean jam',
             'shrimp with chill source','roast chicken','steamed meat dumpling','omelet with fried rice','cutlet curry',
               'spaghetti meat sauce','fried shrimp','potato salad','green salad','macaroni salad','Japanese tofu and vegetable chowder',
               'pork miso soup','chinese soup','beef bowl','kinpira-style sauteed burdock','rice ball',
             'pizza toast','dipping noodles','hot dog','french fries','mixed rice','goya chanpuru']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected.')
    
    file = request.files['file']
       
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type.')
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    prediction = predictImg(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('prediction: ', prediction)

    
    return render_template('uploaded.html', filename=filename, prediction=prediction)


def predictImg(img_path):
    foods = []
    im2 = cv2.imread(img_path)
    results = model.predict(source=im2, save=True, save_txt=True)
    boxes = results[0].boxes    
    for box in boxes:
        foods.append(class_names[int(box.cls)])
    cv2.imwrite(os.path.join('static/images', 'result.jpg'), im2)  # Save image with bounding box to static/images/result.jpg
    return foods
   
if __name__ == '__main__':
    app.run(debug=True)