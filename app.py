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
import random

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
    # data_use = data.iloc[:, [0,1,2,3, 4, 5]].values
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

# creat list values converted
def list_converted(name_colum):  
    global data_use
    dict_converted = {}
    values = list(set(data_use[name_colum]))
    value_replace = 1.0
    for i in range(0, len(values)):
        dict_converted[values[i]] = value_replace
        data_use = data_use.replace(values[i], value_replace)
        value_replace = value_replace +1

    return dict_converted
# reset()
def reset():
    pass
# =========================================DECLARE===========================================================
aler = "Removed data."
aler2 = "Kích thước sau khi remove data:"
list_column_name = list_columns_name()
data_use = get_data_use() 
dic_column_noise = {}
shape = getShape();
accurate_entropy=""
accurate_gini = ""
accurate_bayes = ""
cm_entropy = []
cm_gini = []
cm_bayes = []
filename_converted =""

dict_Road_Class_2 = {} 
dict_Road_Surface_2  = {} 
dict_Lighting_2  = {} 
dict_Weather_2  = {} 
dict_Vehicle_2  = {} 

classifier_entropy = None
classifier_gini = None
classifier_bayes = None
sc = None

# =========================================ROUTE===========================================================
 
#route home -----------------------------------------------------------------------------------------------
@app.route("/")
def index():

    # reset lại data_use
    global data_use 
    data_use = get_data_use()

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

    # reset các giá trị khai báo ban đầu
    reset()

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
    global data_use
    time  = datetime.datetime.now()
    # time_str = str(time)
    time_str = time.strftime('%d_%m_%Y %H_%M_%S')
    data_use.to_csv("./dataset/2017_preprocessed "+time_str+".csv",header = True, index= False)
    return render_template('pre_process.html',  
                            dic_column_noise = dic_column_noise,
                            list_column_name = list_columns_name(),
                            dic_column_missing = dic_column_missing(),
                            shape= shape,  aler2 = aler2, aler= aler)

#route convert values -----------------------------------------------------------------------------------------------
@app.route('/convert_values' )
def convert_values():
    global data_use 

    dict_Road_Class  = {}
    dict_Road_Surface  = {}
    dict_Lighting  = {}
    dict_Weather  = {}
    dict_Vehicle  ={} 

    return render_template('convert_values.html', 
                            data =   data_use.head(),
                            dict_Road_Class = dict_Road_Class,
                            dict_Road_Surface = dict_Road_Surface, 
                            dict_Lighting =dict_Lighting,
                            dict_Weather =dict_Weather ,
                            dict_Vehicle= dict_Vehicle)


 #route converted values -----------------------------------------------------------------------------------------------
@app.route('/converted_values', methods=['POST'] )
def converted_values():
    global data_use
    global dict_Road_Class_2 
    global dict_Road_Surface_2
    global dict_Lighting_2 
    global dict_Weather_2
    global dict_Vehicle_2
        
    dict_Road_Class  = list_converted('1st Road Class & No')
    dict_Road_Surface  = list_converted('Road Surface')
    dict_Lighting  = list_converted('Lighting Conditions')
    dict_Weather  = list_converted('Weather Conditions')
    dict_Vehicle  = list_converted('Type of Vehicle')
 
    # //lưu sang biến khác
    dict_Road_Class_2  = dict_Road_Class
    dict_Road_Surface_2  = dict_Road_Surface
    dict_Lighting_2  = dict_Lighting
    dict_Weather_2  = dict_Weather
    dict_Vehicle_2  = dict_Vehicle
    # kết thúc ham converted dữ liệu trong data_use đã chuyển thành các số
    return render_template('converted_values.html', 
                            data =   data_use.head(),
                            dict_Road_Class = dict_Road_Class,
                            dict_Road_Surface = dict_Road_Surface, 
                            dict_Lighting =dict_Lighting,
                            dict_Weather =dict_Weather ,
                            dict_Vehicle= dict_Vehicle)

@app.route('/save_file_converted', methods=['POST'] )
def save_file_converted():
    # tới đây data_use đã preprogressed , tiến hành lưu csv đã converted, với tên file là filename_converted
    global data_use  
    global filename_converted
    global dict_Road_Class

    dict_Road_Class  = {}
    dict_Road_Surface  = {}
    dict_Lighting  = {}
    dict_Weather  = {}
    dict_Vehicle  = {}
    time  = datetime.datetime.now()
    # time_str = str(time)
    time_str = time.strftime('%d_%m_%Y %H_%M_%S')
    filename_converted = "./dataset/2017_preprocessed_converted"+time_str+".csv"
    data_use.to_csv(filename_converted, header = True, index= False)
    return render_template('converted_values.html',  
                            data =   data_use.head(),
                            dict_Road_Class = dict_Road_Class,
                            dict_Road_Surface = dict_Road_Surface, 
                            dict_Lighting =dict_Lighting,
                            dict_Weather =dict_Weather ,
                            dict_Vehicle= dict_Vehicle)

 #route classify -----------------------------------------------------------------------------------------------
@app.route('/classify')
def classify():  
    global cm_entropy
    global cm_gini
    global data_use

    global dict_Road_Class_2 
    global dict_Road_Surface_2
    global dict_Lighting_2 
    global dict_Weather_2
    global dict_Vehicle_2
        

    return render_template("classification.html", cm_entropy = cm_entropy, cm_gini=cm_gini, cm_bayes = cm_bayes,
                            dict_Road_Class_2 = dict_Road_Class_2,
                            dict_Road_Surface_2= dict_Road_Surface_2,
                            dict_Lighting_2=dict_Lighting_2,
                            dict_Weather_2= dict_Weather_2,
                            dict_Vehicle_2= dict_Vehicle_2) 

# entropy
@app.route('/entropy', methods = ['POST'])
def entropy():
    global accurate_entropy
    global cm_entropy
    global cm_gini
    global filename_converted
    global data_use
    global classifier_entropy
    global cm_bayes
    global sc

    # đọc dữ liệu  filename vừa lưu xúns
    # có thể dùng data_use lun 

    # dataset_entroy = pd.read_csv("./dataset/2017_preprocessed_converted17_12_2018 14_45_58.csv", encoding ='latin1')  

    dataset_entroy = pd.read_csv(filename_converted, encoding ='latin1')  

    # Split data into input and lable output
    X_entropy = dataset_entroy.iloc[:, [0,1,2,3, 4]].values
    y_entropy = dataset_entroy.iloc[:, 5].values

    # Split dataset train, dataset test
    from sklearn.cross_validation import train_test_split
    X_train_entropy, X_test_entropy, y_train_entropy, y_test_entropy = train_test_split(X_entropy, y_entropy, test_size = 0.25,
     random_state = 0)

    # Fit_transform data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_entropy = sc.fit_transform(X_train_entropy)
    X_test_entropy = sc.transform(X_test_entropy)

    #Build model decision tree 
    from sklearn.tree import DecisionTreeClassifier
    classifier_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_entropy.fit(X_train_entropy, y_train_entropy)

    y_pred_entropy = classifier_entropy.predict(X_test_entropy)

    # confussion matrix
    from sklearn.metrics import confusion_matrix
    cm_entropy = confusion_matrix(y_test_entropy, y_pred_entropy) 
    cm_entropy = cm_entropy.tolist();

    # accuracy_score
    from sklearn.metrics import accuracy_score
    accurate_entropy =  accuracy_score(y_test_entropy, y_pred_entropy)

    return render_template("classification.html", accurate_entropy =accurate_entropy,
                            cm_entropy = cm_entropy, cm_gini=cm_gini, cm_bayes = cm_bayes,
                            dict_Road_Class_2 = dict_Road_Class_2,
                            dict_Road_Surface_2= dict_Road_Surface_2,
                            dict_Lighting_2=dict_Lighting_2,
                            dict_Weather_2= dict_Weather_2,
                            dict_Vehicle_2= dict_Vehicle_2)

# Gini
@app.route('/gini', methods = ['POST'])
def gini():
    global accurate_entropy
    global accurate_gini
    global cm_entropy
    global cm_gini
    global cm_bayes
    global data_use
    global classifier_gini
    global sc

    # dataset_gini = pd.read_csv("./dataset/2017_preprocessed 13_12_2018 23_11_37.csv", encoding ='latin1')  
     # Split data into input and lable output

    dataset_gini = pd.read_csv(filename_converted, encoding ='latin1')  
    X_gini = dataset_gini.iloc[:, [0,1,2,3, 4]].values
    y_gini = dataset_gini.iloc[:, 5].values

    # Split dataset train, dataset test
    from sklearn.cross_validation import train_test_split
    X_train_gini, X_test_gini, y_train_gini, y_test_gini = train_test_split(X_gini, y_gini, test_size = 0.25, random_state = 0)

    # Fit_transform data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_gini = sc.fit_transform(X_train_gini)
    X_test_gini = sc.transform(X_test_gini)

    #Build model decision tree 
    from sklearn.tree import DecisionTreeClassifier
    classifier_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    classifier_gini.fit(X_train_gini, y_train_gini)

    y_pred_gini = classifier_gini.predict(X_test_gini)

    # confussion matrix
    from sklearn.metrics import confusion_matrix
    cm_gini = confusion_matrix(y_test_gini, y_pred_gini)
    cm_gini = cm_gini.tolist()

    # accuracy_score
    from sklearn.metrics import accuracy_score
    accurate_gini =  accuracy_score(y_test_gini, y_pred_gini)

    return render_template("classification.html", accurate_gini =accurate_gini, 
                            accurate_entropy=accurate_entropy,  cm_entropy = cm_entropy,cm_gini =cm_gini, cm_bayes = cm_bayes,
                            dict_Road_Class_2 = dict_Road_Class_2,
                            dict_Road_Surface_2= dict_Road_Surface_2,
                            dict_Lighting_2=dict_Lighting_2,
                            dict_Weather_2= dict_Weather_2,
                            dict_Vehicle_2= dict_Vehicle_2)

# Bayes
@app.route('/bayes', methods = ['POST'])
def bayes():
    global accurate_entropy
    global accurate_gini
    global accurate_bayes
    global cm_entropy
    global cm_gini
    global cm_bayes
    global data_use
    global classifier_bayes
    global sc

    # dataset_bayes = pd.read_csv("./dataset/2017_preprocessed 13_12_2018 23_11_37.csv", encoding ='latin1')  
     # Split data into input and lable output

    dataset_bayes = pd.read_csv(filename_converted)  
    X_bayes = dataset_bayes.iloc[:, [0,1,2,3, 4]].values
    y_bayes = dataset_bayes.iloc[:, 5].values

    # Split dataset train, dataset test
    from sklearn.cross_validation import train_test_split
    X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = train_test_split(X_bayes, y_bayes, test_size = 0.25, random_state = 0)

    # Fit_transform data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_bayes = sc.fit_transform(X_train_bayes)
    X_test_bayes = sc.transform(X_test_bayes)

    #Build model decision tree 
    from sklearn.naive_bayes import GaussianNB
    classifier_bayes = GaussianNB() 
    classifier_bayes.fit(X_train_bayes, y_train_bayes)

    y_pred_bayes = classifier_bayes.predict(X_test_bayes)

    # confussion matrix
    from sklearn.metrics import confusion_matrix
    cm_bayes = confusion_matrix(y_test_bayes, y_pred_bayes)
    cm_bayes = cm_bayes.tolist()

    # accuracy_score
    from sklearn.metrics import accuracy_score
    accurate_bayes =  accuracy_score(y_test_bayes, y_pred_bayes)

    return render_template("classification.html", accurate_gini =accurate_gini, 
                            accurate_entropy=accurate_entropy,  
                            accurate_bayes =accurate_bayes,
                            cm_entropy = cm_entropy,cm_gini =cm_gini, cm_bayes = cm_bayes,
                            dict_Road_Class_2 = dict_Road_Class_2,
                            dict_Road_Surface_2= dict_Road_Surface_2,
                            dict_Lighting_2=dict_Lighting_2,
                            dict_Weather_2= dict_Weather_2,
                            dict_Vehicle_2= dict_Vehicle_2)


#classified 
@app.route('/classified', methods = ['POST'])
def classified():
    # global accurate_entropy
    # global accurate_gini
    global classifier_entropy
    global classifier_gini
    global classifier_bayes
    global sc

    if request.method == 'POST':
        x_test = [None]*5
        x_test[0] = request.form['cot1']
        x_test[1] = request.form['cot2']
        x_test[2] = request.form['cot3']
        x_test[3] = request.form['cot4']
        x_test[4] = request.form['cot5'] 

        import numpy as np
        x_test_2 = np.array([ x_test[0], x_test[1], x_test[2], x_test[3], x_test[4]]) 
        x_test_2D = [x_test_2]


        # # thử fit_transform với dữ liêu 
        # dataset_bayes = pd.read_csv(filename_converted)  
        # X_bayes = dataset_bayes.iloc[:, [0,1,2,3, 4]].values
        # y_bayes = dataset_bayes.iloc[:, 5].values

        # # Split dataset train, dataset test
        # from sklearn.cross_validation import train_test_split
        # X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = train_test_split(X_bayes, y_bayes, test_size = 0.25, random_state = 0)

        # # Fit_transform data
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X_train_bayes = sc.fit_transform(X_train_bayes)
        # X_test_bayes = sc.transform(X_test_bayes)

        # transform 
        x_test_2D_1 = sc.transform(x_test_2D) 

        name = request.form['action']

        if(name == "Entropy"):
            y_pred = classifier_entropy.predict(x_test_2D_1)
        elif (name == "Gini"):
            y_pred = classifier_gini.predict(x_test_2D_1)
        elif (name == "Bayes"):
            y_pred = classifier_bayes.predict(x_test_2D_1) 

    return render_template("classification.html", accurate_gini =accurate_gini, 
                            accurate_entropy=accurate_entropy, accurate_bayes = accurate_bayes,
                            cm_entropy = cm_entropy,cm_gini =cm_gini, cm_bayes = cm_bayes,
                            dict_Road_Class_2 = dict_Road_Class_2,
                            dict_Road_Surface_2= dict_Road_Surface_2,
                            dict_Lighting_2=dict_Lighting_2,
                            dict_Weather_2= dict_Weather_2,
                            dict_Vehicle_2= dict_Vehicle_2, x_test_list=x_test , y_pred = y_pred, name =name)

if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
