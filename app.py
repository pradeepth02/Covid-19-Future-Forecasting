from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np

 
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from flask import Flask, render_template, request, send_from_directory
import utils
import train_models as tm
import os
import pandas as pd
 

app = Flask(__name__) #Initialize the flask App

 
pol = pickle.load(open('polyss.pkl','rb'))
regresso = pickle.load(open('regresoss.pkl','rb'))
scaler = MinMaxScaler()

def perform_training(stock_name, df, models_list):
    all_colors = {'SVR_linear': '#FF9EDD',
                  'SVR_poly': '#FFFD7F',
                  'SVR_rbf': '#FFA646',
                  'linear_regression': '#CC2A1E',
                  'random_forests': '#8F0099',
                  'KNN': '#CCAB43',
                  'elastic_net': '#CFAC43',
                  'DT': '#85CC43',
                  'LSTM_model': '#CC7674'}

    print(df.head())
    dates, prices, ml_models_outputs, prediction_date, test_price = tm.train_predict_plot(stock_name, df, models_list)
    origdates = dates
    if len(dates) > 20:
        dates = dates[-20:]
        prices = prices[-20:]

    all_data = []
    all_data.append((prices, 'false', 'Data', '#000000'))
    for model_output in ml_models_outputs:
        if len(origdates) > 20:
            all_data.append(
                (((ml_models_outputs[model_output])[0])[-20:], "true", model_output, all_colors[model_output]))
        else:
            all_data.append(
                (((ml_models_outputs[model_output])[0]), "true", model_output, all_colors[model_output]))

    all_prediction_data = []
    all_test_evaluations = []
    all_prediction_data.append(("Original", test_price))
    for model_output in ml_models_outputs:
        all_prediction_data.append((model_output, (ml_models_outputs[model_output])[1]))
        all_test_evaluations.append((model_output, (ml_models_outputs[model_output])[2]))

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations

all_files = utils.read_all_stock_files('individual_country_1yr')
 










@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/chart')
def chart():
    return render_template('chart.html')    
    
@app.route('/abstract')
def abstract():
    return render_template('abstract.html')
@app.route('/future')
def future():
    return render_template('future.html')    
    
@app.route('/index2')
def index2():
    return render_template('index2.html')
    
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 

@app.route('/landing_function')
 
def landing_function():
     
    stock_files = list(all_files.keys())

    return render_template('index.html',show_results="false", stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[],
                           prediction_date="", dates=[], all_data=[], len=len([]))

@app.route('/process', methods=['POST'])
def process():

    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')

    
    df = all_files[str(stock_file_name)]
    
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = perform_training(str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())

    return render_template('index.html',all_test_evaluations=all_test_evaluations, show_results="true", stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data))
        
 
@app.route('/index3')
def index3():
 	return render_template("index3.html")
    
@app.route('/predict',methods=['POST'])
def predict():
	int_feature = [x for x in request.form.values()]
	 
	final_features = [np.array(int_feature)]
	Total_infections  = pol.transform(final_features)
	prediction=regresso.predict(Total_infections )
	pred=format(int(prediction[0]))
	
	return render_template('index3.html', prediction_text= pred)    

 
 

if __name__=='__main__':
    app.run(debug=True)
