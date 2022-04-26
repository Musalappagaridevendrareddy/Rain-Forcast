from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle


# def Yield_Pred(List):
# 	import pickle
# 	import numpy as np
# 	arr33 = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
# 			 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
# 			 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
# 			 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
# 			 'Central African Republic', 'Chile', 'Colombia', 'Croatia',
# 			 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
# 			 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
# 			 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
# 			 'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
# 			 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
# 			 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
# 			 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
# 			 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
# 			 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
# 			 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
# 			 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
# 			 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
# 			 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
# 			 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
# 			 'Uruguay', 'Zambia', 'Zimbabwe']
#
# 	arr22 = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
# 			 'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']
#
# 	arr2 = []
# 	for i in arr22:
# 		arr2.append('Season_' + i)
#
# 	arr3 = []
# 	for i in arr33:
# 		arr3.append('Crop_' + i)
#
# 	array_zero = np.zeros(len(arr2))
# 	df_new2 = pd.DataFrame(array_zero)
# 	df_new2 = df_new2.T
# 	df_new2.columns = arr2
#
# 	array_zero = np.zeros(len(arr3))
# 	df_new3 = pd.DataFrame(array_zero)
# 	df_new3 = df_new3.T
# 	df_new3.columns = arr3
#
# 	input = List
#
# 	s3 = input[3]
# 	s2 = input[4]
#
# 	s2 = 'Country_' + s2
# 	s3 = 'Item_' + s3
#
# 	def converting_to_onehot(s, df, arr):
# 		df[s] = 1
#
# 		# df.drop(s,axis=1,inplace=True)
# 		# fill every other column with 0
# 		for i in arr:
# 			if i != s:
# 				df[i] = 0
# 		return df
#
# 	io = [1485.0, 121.00, 16.37, s2, s3]
# 	df2 = converting_to_onehot(input[3], df_new2, arr2)
# 	df3 = converting_to_onehot(input[4], df_new3, arr3)
#
# 	df_pre = pd.concat([df2, df3], axis=1)
# 	df_pre['average_rain_fall_mm_per_year	'] = io[0]
# 	# df_pre['pesticides_tonnes']=io[1]
# 	df_pre['avg_temp'] = io[2]
# 	# df_pre['percent_of_production']=io[1]
# 	model_yield = pickle.load(open("./models/finalized_model.sav", "rb"))
# 	return model_yield.predict(df_pre)
#
app = Flask(__name__, template_folder="template")
# model = pickle.load(open("./models/cat.pkl", "rb"))
# print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")


# @app.route("/api_predict",methods=['POST'])
# @cross_origin()
# def api_predict():
# 	List = request.json['list']
# 	pred = model.predict(List)
# 	return jsonify({'output':str(pred)})
#
#
# @app.route("/api_predict_yield",methods=['POST'])
# @cross_origin()
# def api_predict_yield():
# 	List = request.json['list']
# 	pred = Yield_Pred(List)
# 	return jsonify({'output':str(pred[0])})

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		# DATE
		date = request.form['date']
		day = float(pd.to_datetime(date, format="%Y-%m-%dT").day)
		month = float(pd.to_datetime(date, format="%Y-%m-%dT").month)
		# MinTemp
		minTemp = float(request.form['mintemp'])
		# MaxTemp
		maxTemp = float(request.form['maxtemp'])
		# Rainfall
		rainfall = float(request.form['rainfall'])
		# Evaporation
		evaporation = float(request.form['evaporation'])
		# Sunshine
		sunshine = float(request.form['sunshine'])
		# Wind Gust Speed
		windGustSpeed = float(request.form['windgustspeed'])
		# Wind Speed 9am
		windSpeed9am = float(request.form['windspeed9am'])
		# Wind Speed 3pm
		windSpeed3pm = float(request.form['windspeed3pm'])
		# Humidity 9am
		humidity9am = float(request.form['humidity9am'])
		# Humidity 3pm
		humidity3pm = float(request.form['humidity3pm'])
		# Pressure 9am
		pressure9am = float(request.form['pressure9am'])
		# Pressure 3pm
		pressure3pm = float(request.form['pressure3pm'])
		# Temperature 9am
		temp9am = float(request.form['temp9am'])
		# Temperature 3pm
		temp3pm = float(request.form['temp3pm'])
		# Cloud 9am
		cloud9am = float(request.form['cloud9am'])
		# Cloud 3pm
		cloud3pm = float(request.form['cloud3pm'])
		# Cloud 3pm
		location = float(request.form['location'])
		# Wind Dir 9am
		winddDir9am = float(request.form['winddir9am'])
		# Wind Dir 3pm
		winddDir3pm = float(request.form['winddir3pm'])
		# Wind Gust Dir
		windGustDir = float(request.form['windgustdir'])
		# Rain Today
		rainToday = float(request.form['raintoday'])

		input_lst = [location , minTemp , maxTemp , rainfall , evaporation , sunshine ,
					 windGustDir , windGustSpeed , winddDir9am , winddDir3pm , windSpeed9am , windSpeed3pm ,
					 humidity9am , humidity3pm , pressure9am , pressure3pm , cloud9am , cloud3pm , temp9am , temp3pm ,
					 rainToday , month , day]
		print(input_lst)
		pred = model.predict(input_lst)
		output = pred
		if output == 0:
			return render_template("after_sunny.html")
		else:
			return render_template("after_rainy.html")
	return render_template("predictor.html")


if __name__=='__main__':
	import os
	port = int(os.environ.get('PORT', 8080))
	app.run(host='0.0.0.0', port=port)
	# app.run(debug=True)
