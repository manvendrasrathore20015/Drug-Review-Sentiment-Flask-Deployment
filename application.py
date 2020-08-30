from flask import Flask, render_template,url_for, request
from werkzeug.utils import secure_filename
import csv
import pickle
import flask_monitoringdashboard as dashboard
import warnings
import os
from flask_cors import CORS, cross_origin
from applicaton_logging import logger
from trainingModel import trainModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def warns(*args, **kwargs):
    pass
warnings.warn = warns

# load the model from directory
filename = 'pickle_files/drug_LinearSVC.pkl'
model = pickle.load(open(filename, 'rb'))
t = pickle.load(open('pickle_files/d_transform.pkl', 'rb'))

app = Flask(__name__)
# for monitoring
dashboard.bind(app)
# --- Cross Origin Resource Sharing (CORS) ---
CORS(app)

#logging object initialization
logger = logger.App_Logger()

@app.route('/')
@cross_origin()
def home():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= Home Page Opened =============')
    file_object.close()
    return render_template('home.html')

@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= Bulk Prediction Started =============')
    if request.method == "POST":
        try:
            f = request.files['csvfile']
            logger.log(file_object, 'File submitted for bulk prediction')
            if f:
                f.save(secure_filename(f.filename))
                logger.log(file_object, 'File saved to directory')
                try:
                    with open(f.filename, encoding='Latin1') as file:
                        csvfile = csv.reader(file)
                        data = []
                        review_prediction = []
                        for row in csvfile:
                            data.append(row)
                except Exception as e:
                    os.remove(f.filename)
                    logger.log(file_object,"File uploded is not csv ..")

                for review in data:
                    review_prediction.append(model.predict(t.transform(review)))
                logger.log(file_object, 'Data passed to model ')
                length = len(data)
                os.remove(f.filename)
                logger.log(file_object, 'Saved file removed successfully')
                file.close()
                logger.log(file_object, '============= Bulk Prediction Complete =============')
                file_object.close()
                return render_template("bulk.html", predict_data=review_prediction, data=data, length=length)
        except Exception as e:
            logger.log(file_object, 'Bulk Upload Failed . ERROR message :  ' + str(e))
            file_object.close()
            return "File uploded should be be csv (.csv extension) "

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    try:
        if request.method == 'POST':
            logger.log(file_object, '============= Single Prediction Started =============')
            message = request.form['message']
            logger.log(file_object, 'Data taken for single prediction')
            data = [message]
            my_prediction = model.predict(t.transform(data))
            logger.log(file_object, 'Data passed to model for prediction ')
            logger.log(file_object, '============= Single Prediction Completed =============')
            file_object.close()
        return render_template('result.html',prediction=my_prediction)
    except Exception as e:
        logger.log(file_object, 'Single Prediction Failed . ERROR message :  '+str(e))
        file_object.close()
        return 'Something went wrong'

@app.route('/about', methods=['POST'])
@cross_origin()
def about():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    logger.log(file_object, '============= About Page Opened =============')
    if request.method == 'POST':
        logger.log(file_object, 'Returning about page')
        file_object.close()
        return render_template('about.html')

@app.route('/retrain',methods=['GET','POST'])
@cross_origin()
def retrain():
    file_object = open("log_file/FlaskApi_log.txt", 'a+')
    try:
        logger.log(file_object, '============= Retraining Model Started =============')
        if request.method == "POST":
            file = request.files['retrain_file']
            if file:
                file.save(secure_filename(file.filename))
                a=trainModel()
                a.trainingModel(file.filename,file_object)
                logger.log(file_object, '============= Model Retraining Done =============')
                os.remove(file.filename)
                file_object.close()
                return render_template('home.html',text=".... Model Retrained Successfully ....")
    except Exception as e:
        logger.log(file_object, 'Model Retraining Failed . ERROR message :  ' + str(e))
        file_object.close()
        return 'Something went wrong , check your file extension .(should be .csv )'

if __name__ == '__main__':
    # To run on web ..
    #app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run(debug=True)