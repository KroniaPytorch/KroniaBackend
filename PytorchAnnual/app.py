from flask import Flask, jsonify, request, render_template, redirect, url_for,send_file 
from flask_api import FlaskAPI, status, exceptions
import os, json, requests
from werkzeug.utils import secure_filename
from requests import get
import Crop, Fertilizer, Wheatdisease, FruitRecommender, LeafDisease, YellowMosaic, Cotton, Soil
from dotenv import load_dotenv
load_dotenv()

UPLOAD_FOLDER = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

api_endpoint=f""+os.getenv('api_endpoint')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/') 
def index(): 
	return "Flask server"

@app.route("/getCoord", methods=["GET"])
def get_my_ip():
    # ip_addr=request.environ['REMOTE_ADDR']
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        print(request.environ['REMOTE_ADDR'])
        ip_addr=request.environ['REMOTE_ADDR']
    else:
        print(request.environ['HTTP_X_FORWARDED_FOR'])
        ip_addr=request.environ['HTTP_X_FORWARDED_FOR']
        
    # ip_addr=request.remote_addr
    print(ip_addr)
    ip = '208.67.222.22'

    latlong = get('https://ipapi.co/'+ip_addr+'/latlong/'.format(ip_addr)).text.split(',')

    # weather = get('http://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid=API_KEY'.format(latlong[0], latlong[1])).json()

    print(latlong)
    return jsonify({'msg': 'success','coordinates':latlong}) 
    


###############################Filter API################################################
@app.route('/getPlantNet', methods = ['GET','POST']) 
def filter():
    img_data = request.files['image']
    value=request.form['value']
    if value not in ['leaf','fruit','flower','bark']:
        return jsonify({'Error':'Invalid value. Value shoudl match fruit,flower,leaf or bark'})
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    result="File Not Allowed"
    json_msg=''

    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        dataa= open(img_path, 'rb')
        data = {	'organs': [value]}
        files = [('images', (img_path, dataa))]
        req = requests.Request('POST', url=api_endpoint, files=files, data=data)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared)
        json_result = json.loads(response.text)
        # print(json_result)
        if 'message' not in json_result:
            json_msg='Valid Image'
        else:
            json_msg='Invalid Image'
       
        try:
            os.remove(img_path)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':json_msg}) 


###################################################CROP RECOMMENDER#################################################    

@app.route('/getSoil', methods = ['GET','POST']) 
def soil():
    
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})
    f=0
    result=""
    Error="None"
    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= Soil.inference(img_path)
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error}) 

@app.route('/getCrop', methods = ['GET','POST']) 
def cropRecommender():
    req_data = request.get_json()
    result= Crop.predict(req_data)
    return jsonify({'msg': 'success','result':result})

@app.route('/getFertilizer', methods = ['GET','POST']) 
def fertilizerRecommender():
    req_data = request.get_json()
    result= Fertilizer.predict(req_data)
    return jsonify({'msg': 'success','result':result})

######################################################FRUIT RECOMMENDER ###################################################3    

@app.route('/getFruitRecommender', methods = ['GET','POST']) 
def fruitRecommender():
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    result="File Not Allowed"
    f=0
    result=""
    Error="None"
    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= FruitRecommender.inference(img_path,'train_annot.csv','Fruit_Recommender.pth')
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error}) 



######################################################COTTON DISEASE###################################################3    

@app.route('/getCotton', methods = ['GET','POST']) 
def cotton():
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    f=0
    result=""
    Error="None"

    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= Cotton.inference(img_path)
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error})  

######################################################Yello Mosaic DISEASE###################################################3    

@app.route('/getYellow', methods = ['GET','POST']) 
def yellow():
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    f=0
    result=""
    Error="None"

    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= YellowMosaic.inference(img_path)
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error})  

######################################################LEAF DISEASE###################################################3    


@app.route('/getLeafDisease', methods = ['GET','POST']) 
def leafDisease():
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    f=0
    result=""
    Error="None"

    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= LeafDisease.inference(img_path)
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error})  

######################################################WHEAT DISEASE###################################################3    

@app.route('/getWheatDisease', methods = ['GET','POST']) 
def wheatDisease():
    img_data = request.files['image']
    if img_data.filename == '':
        return jsonify({'msg':"Image not found"})

    f=0
    result=""
    Error="None"

    if img_data and allowed_file(img_data.filename):
        filename = secure_filename(img_data.filename)
        img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(img_path)
        img_data.save(img_path)
        try:
            result= Wheatdisease.predict(img_path)
        except Exception as error:
            Error="Error in the model Inference File"
            try:
                f=1
                os.remove(img_path)
            except Exception as error2:
                Error="Error removing or closing downloaded file handle and buggy inference file"
                app.logger.error("Error removing or closing downloaded file handle", error2)
            app.logger.error("Error in the model Inference File", error)
        if f==0:
            try:
                os.remove(img_path)
            except Exception as error:
                Error="Error removing or closing downloaded file handle"
                app.logger.error("Error removing or closing downloaded file handle", error)
    return jsonify({'msg': 'success','result':result,'error':Error}) 
    return jsonify({'msg': 'success','result':result})



if __name__ == "__main__": 
	app.run(threaded=True, port = int(os.environ.get('PORT', 5000)))



# soil_npk={'Black':{'Nitrogen':12,'Phosporous'}}