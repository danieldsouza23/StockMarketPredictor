import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def get_data(filename):
    price = []
    date = []
    file = filename +'.csv'
    stock = pd.read_csv(file)
    iters = len(stock)
    for i in range(0,iters):
        tdate = int(stock['Date'][i].split('-')[0])
        date.append(tdate)
        tprice = float(stock['Open'][i])
        price.append(tprice)
    price.reverse()
    date.reverse()
    return price,date

def get_price(price,dates,x):
    datex = []
    datex = np.reshape(date,(len(date),1))

    svr_lin = SVR(kernel ='linear', C =1e3)
    svr_pol = SVR(kernel ='poly',   C =1e3, degree =2)
    svr_rbf = SVR(kernel ='rbf',    C=1e3,  gamma = 0.1)

    svr_lin.fit(datex,price)
    svr_pol.fit(datex,price)
    svr_rbf.fit(datex,price)

    plt.scatter(datex,price, color='black', label='Data')
    plt.plot(datex, svr_lin.predict(datex), color = 'orange', label ='Linear SVR')
    plt.plot(datex, svr_pol.predict(datex), color = 'blue',   label = 'Polynomial SVR')
    plt.plot(datex, svr_rbf.predict(datex), color = 'green',  label = 'Radio Base Func SVR')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_pol.predict(x)[0], svr_rbf.predict(x)[0]
