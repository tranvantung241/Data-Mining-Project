import os
# from django.shortcuts import redirect
from flask import Flask, request, render_template, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys   
from flask import render_template_string 
from wtforms import Form, TextField 
import datetime

__author__ = 'TranTung'
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
 

# =========================================FUNCTION===========================================================
# read data
def read_data():
    data = pd.read_csv("./dataset/2017.csv", encoding ='latin1')  
    return data

# renew data
def renew_data(data_handed):
    return data_handed


# get data  use]
def get_data_use():
    data = read_data()
    data_use = data[['1st Road Class & No','Road Surface','Lighting Conditions','Weather Conditions','Type of Vehicle', 'Casualty Severity']]
    return data_use

# hàm gọi các tên column
def list_columns_name():
    data_use = get_data_use()
    list_column_name = data_use.columns 
    return list_column_name

# get shape data use
def getShape():
    data_use = get_data_use()
    shape = data_use.shape
    return shape

# khởi tạo dictionary rỗng
def dic_column_missing():
    dic_column_missing = {}
    return dic_column_missing

# =========================================DECLARE===========================================================
aler = "Removed data."
aler2 = "Kích thước sau khi remove data:"
list_column_name = list_columns_name()
data_use = get_data_use() 
dic_column_noise = {}
shape = getShape();

# =========================================ROUTE===========================================================
#route home -----------------------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")


#route describle -----------------------------------------------------------------------------------------------
@app.route("/describle")
def describle():
    return render_template("describle.html")

#route upload_file -----------------------------------------------------------------------------------------------
@app.route("/upload_file")
def upload_file():
    return render_template("upload.html")

#route upload -----------------------------------------------------------------------------------------------
@app.route("/upload", methods=['POST', 'GET'])
def upload(): 
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if ext in [".jpg", ".png", ".csv"]: 
        # if (ext == ".jpg") or (ext == ".png") or (ext == ".csv"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination) 
    return render_template("complete.html")


 
#route showw data -----------------------------------------------------------------------------------------------
@app.route('/show')
def show(): 
    return render_template("show.html", df_view = read_data())

#route show data use -----------------------------------------------------------------------------------------------
@app.route('/show_use')
def show_use(): 
    return render_template("show_use.html", df_view = get_data_use())

#route chart -----------------------------------------------------------------------------------------------
@app.route('/chart')
def chart():
    return render_template("chart.html")

#route preprocess -----------------------------------------------------------------------------------------------
@app.route('/preprocess')
def preprocess():  
    return render_template("pre_process.html", list_column_name = list_columns_name(),
                             dic_column_missing = dic_column_missing(), 
                             len=len(list_columns_name()), shape= getShape(), dic_column_noise =dic_column_noise)


#route check_missing -----------------------------------------------------------------------------------------------
@app.route('/check_missing', methods=['POST'] )
def check_missing():  
    global data_use
    dic_column_missing = {}
    sum_missing = []
    # nhận dữ liệu từ form submit
    if request.method == 'POST':
        for i in range(0, len(list_column_name)):
            sum = data_use[list_column_name[i]].isnull().sum()
            dic_column_missing[list_column_name[i]] = sum
    return render_template('pre_process.html',
                            dic_column_noise = dic_column_noise,
                            dic_column_missing = dic_column_missing, 
                            list_column_name = list_columns_name(),
                            shape= getShape())

#route Remove missing -----------------------------------------------------------------------------------------------
@app.route('/remove_missing', methods=['POST'] )
def remove_missing():  
    global data_use
    global shape
    dic_column_missing = {}
    sum_missing = []
    # nhận dữ liệu từ form submit
    if request.method == 'POST':
         data_use = data_use.dropna()  
         shape = data_use.shape
    return render_template('pre_process.html',
                            dic_column_noise = dic_column_noise,
                            dic_column_missing = dic_column_missing, 
                            list_column_name = list_columns_name(),
                            shape= shape,  
                            aler =aler,
                            aler2 = aler2)

#route check_Noise -----------------------------------------------------------------------------------------------
@app.route('/check_noise', methods=['POST'] )
def check_noise():    
    global shape
    number  = data_use['1st Road Class & No'].value_counts()
    index_number =  number.keys()
    dic_column_noise = {}

    if request.method == 'POST':
        for i in range(0, len(number)): 
            dic_column_noise[index_number[i]] = number[i]

    # # vẽ hiểu đồ
    # import matplotlib.pyplot as plt
    # import plotly.plotly as py
    # import plotly.tools as tls
    # plt.rcParams['figure.figsize'] = (40, 24)
    # plt.bar(index_number,number, align='center', alpha=0.5)
    # plt.savefig('./static/image/noise.jpg')


    return render_template('pre_process.html',  
                            dic_column_noise = dic_column_noise,
                            list_column_name = list_columns_name(),
                            dic_column_missing = dic_column_missing(),
                            shape= shape
                            )
#route remove_Noise -----------------------------------------------------------------------------------------------
@app.route('/remove_noise', methods=['POST'] )
def remove_noise():
    global data_use;
    global shape;

    # xóa cột  1st Road Class & No
    remove_list = []
    number  = data_use['1st Road Class & No'].value_counts()
    index_number =  number.keys()
    for i in range(0, len(number)):
        if (number[i]<20):
            remove_list.append(index_number[i])

    for i in remove_list:
        data_use = data_use[data_use['1st Road Class & No'] != i]


    # xóa cột Weather Conditions   
    data_use = data_use[data_use['Weather Conditions'] != 'Other']


    shape = data_use.shape

    return render_template('pre_process.html',  
                            dic_column_noise = dic_column_noise,
                            list_column_name = list_columns_name(),
                            dic_column_missing = dic_column_missing(),
                            shape= shape,  aler2 = aler2, aler= aler)


@app.route('/save_file', methods=['POST'] )
def save_file():
    time  = datetime.datetime.now()
    # time_str = str(time)
    time_str = time.strftime('%d_%m_%Y %H_%M_%S')
    data_use.to_csv("./dataset/2017_preprocessed "+time_str+".csv",header = True, index= False)
    return render_template('pre_process.html',  
                            dic_column_noise = dic_column_noise,
                            list_column_name = list_columns_name(),
                            dic_column_missing = dic_column_missing(),
                            shape= shape,  aler2 = aler2, aler= aler)

if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
