from flask import Flask, render_template
from config import DevConfig
import numpy as np
#import matplotlib.pyplot as plt 機器學習計算
import csv
import math
list2010 = []
list2011 = []
list2012 = []
list2013 = []
list2014 = []
list2015 = []

with open('C0ACA0-2010-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2010.append(row)

with open('C0ACA0-2011-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2011.append(row)
		
with open('C0ACA0-2012-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2012.append(row)

with open('C0ACA0-2013-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2013.append(row)

with open('C0ACA0-2014-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2014.append(row)

with open('C0ACA0-2015-03.csv' , newline = '') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows :
		list2015.append(row)

b=-4
w1=0.1
w2=0.1
w3=0.1
w4=0.1
w5=0.1
w6=0.1
w7=0.1

lr=1                
iteration=1000    

b_history=[b]       
w1_history=[w1]       
w2_history=[w2]
w3_history=[w3]       
w4_history=[w4]
w5_history=[w5]       
w6_history=[w6]
w7_history=[w7]

lr_b=0
lr_w1=0
lr_w2=0
lr_w3=0
lr_w4=0
lr_w5=0
lr_w6=0
lr_w7=0

#training
for i in range(iteration):
    b_grad=0.0  
    w1_grad=0.0  
    w2_grad=0.0
    w3_grad=0.0  
    w4_grad=0.0
    w5_grad=0.0
    w6_grad=0.0
    w7_grad=0.0
    iter = 0
    for n in range(len(list2010)-1):
        iter = iter +1
        b_grad = b_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*1.0
        w1_grad = w1_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2010[iter][7])
        w2_grad = w2_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2011[iter][7])
        w3_grad = w3_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2012[iter][7])
        w4_grad = w4_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2013[iter][7])
        w5_grad = w5_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2014[iter][7])
        w6_grad = w6_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*float(list2014[iter][0])
        w7_grad = w7_grad -2.0*(float(list2015[iter][7]) - b - w1*float(list2010[iter][7])-w2*float(list2011[iter][7]) - w3*float(list2012[iter][7])-w4*float(list2013[iter][7]) - w5*float(list2014[iter][7]) - w6*float(list2014[iter][0]) - w7*(float(list2014[iter][0])**2))*(float(list2014[iter][0])**2)

    lr_b = lr_b + b_grad **2
    lr_w1 = lr_w1 + w1_grad **2
    lr_w2 = lr_w2 + w2_grad **2
    lr_w3 = lr_w3 + w3_grad **2
    lr_w4 = lr_w4 + w4_grad **2
    lr_w5 = lr_w5 + w5_grad **2
    lr_w6 = lr_w6 + w6_grad **2
    lr_w7 = lr_w7 + w7_grad **2
	
    b = b - lr/np.sqrt(lr_b)*b_grad # Adagrad
    w1 = w1 - lr/np.sqrt(lr_w1)*w1_grad
    w2 = w2 - lr/np.sqrt(lr_w2)*w2_grad
    w3 = w3 - lr/np.sqrt(lr_w3)*w3_grad
    w4 = w4 - lr/np.sqrt(lr_w4)*w4_grad
    w5 = w5 - lr/np.sqrt(lr_w5)*w5_grad
    w6 = w6 - lr/np.sqrt(lr_w6)*w6_grad
    w7 = w7 - lr/np.sqrt(lr_w7)*w7_grad

    b_history.append(b)
    w1_history.append(w1)
    w2_history.append(w2)
    w3_history.append(w3)
    w4_history.append(w4)
    w5_history.append(w5)
    w6_history.append(w6)
    w7_history.append(w7)

        
print("Mini point: b = %.4f" %b_history[-1],"w1 = %.4f" % w1_history[-1] ,"w2 = %.4f" % w2_history[-1], "w3 = %.4f " % w3_history[-1] ,"w4 = %.4f" % w4_history[-1]," w5 = %.4f" % w5_history[-1] , " w6 = %.4f" % w6_history[-1] , " w7 = %.4f" %w7_history[-1])
iter = 0
MAE = 0
for n in range(len(list2010)-1):
    iter = iter+1
    predicted = b_history[-1] + float(w1_history[-1]) * float(list2010[iter][7]) + float(w2_history[-1]) * float(list2011[iter][7]) + float(w3_history[-1]) * float(list2012[iter][7]) + float(w4_history[-1]) * float(list2013[iter][7]) + float(w5_history[-1]) * float(list2014[iter][7]) + float(w6_history[-1]) * float(list2014[iter][0]) + float(w7_history[-1]) * (float(list2014[iter][0])**2)
    actual = float(list2015[iter][7])
    print("Date : ",list2014[iter][0] ,"The predicted temperature : %.1f" % (predicted) , "     The actual temperature : " , actual)
    MAE = MAE + math.fabs(predicted-actual)
MAE = MAE/31
print("Mean absolute error = %.2f" % (MAE))
# add  flask 架網站
app = Flask(__name__)
app.config.from_object(DevConfig)

@app.route('/')
def index():
	data = {'b':b,'w1':w1,'w2':w2,'w3':w3,'w4':w4,'w5':w5,'w6':w6,'w7':w7}; #傳值到html
	return render_template("front.html",data=data);

if __name__ == '__main__':
    app.run(debug = True);