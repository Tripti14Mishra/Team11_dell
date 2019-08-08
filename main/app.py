from flask import Flask, render_template, request,jsonify
import json

app = Flask(__name__)



# -----------------------------------------------------------------------------------------------------
import pandas as pd
data = pd.read_csv("ewerdata.csv")
data.shape
import random

df = {'user_name':[],'laptop_name':[],'searched':[],'company':[],'typename':[],'inches':[],'resolution':[],'cpu':[],'ram':[],'memory':[],'gpu':[],'os':[],'weight':[],'price':[]}
for i in range(1,40):
    for j in range(10):
        s = "user_"+str(i)
        df['user_name'].append(s)
        rarray = random.sample(range(1, 21), 10)
        df['laptop_name'].append(data['Product'][rarray[j]])
        df['searched'].append(random.randint(1,4))
        df['company'].append(data['Company'][rarray[j]])
        df['typename'].append(data['TypeName'][rarray[j]])
        df['inches'].append(data['Inches'][rarray[j]])
        df['resolution'].append(data['ScreenResolution'][rarray[j]])
        df['cpu'].append(data['Cpu'][rarray[j]])
        df['ram'].append(data['Ram'][rarray[j]])
        df['memory'].append(data['Memory'][rarray[j]])
        df['gpu'].append(data['Gpu'][rarray[j]])
        df['os'].append(data['OpSys'][rarray[j]])
        df['weight'].append(data['Weight'][rarray[j]])
        df['price'].append(data['Price_euros'][rarray[j]])


newdata = pd.DataFrame(df)
# %matplotlib inline

import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib

mat = newdata.pivot_table(index='user_name', columns='laptop_name', values='searched')

#Company	Product	TypeName	Inches	ScreenResolution	Cpu	Ram	Memory	Gpu	OpSys	Weight	Price_euros
pddata = data.copy()

from sklearn import preprocessing
lecompany = preprocessing.LabelEncoder()
pddata['Company'] = lecompany.fit_transform(pddata['Company'])

leproduct = preprocessing.LabelEncoder()
pddata['Product'] = leproduct.fit_transform(pddata['Product'])

letypename = preprocessing.LabelEncoder()
pddata['TypeName'] = letypename.fit_transform(pddata['TypeName'])

leinches = preprocessing.LabelEncoder()
pddata['Inches'] = leinches.fit_transform(pddata['Inches'])

lescreenresolution = preprocessing.LabelEncoder()
pddata['ScreenResolution'] = lescreenresolution.fit_transform(pddata['ScreenResolution'])

lecpu = preprocessing.LabelEncoder()
pddata['Cpu'] = lecpu.fit_transform(pddata['Cpu'])

leram = preprocessing.LabelEncoder()
pddata['Ram'] = leram.fit_transform(pddata['Ram'])

lememory = preprocessing.LabelEncoder()
pddata['Memory'] = lememory.fit_transform(pddata['Memory'])

legpu = preprocessing.LabelEncoder()
pddata['Gpu'] = legpu.fit_transform(pddata['Gpu'])

leopsys = preprocessing.LabelEncoder()
pddata['OpSys'] = leopsys.fit_transform(pddata['OpSys'])

leweight = preprocessing.LabelEncoder()
pddata['Weight'] = leweight.fit_transform(pddata['Weight'])

leprice_euros = preprocessing.LabelEncoder()
pddata['Price_euros'] = leprice_euros.fit_transform(pddata['Price_euros'])


y = pddata.iloc[:,2]
pddata = pddata.drop(columns=['Product'])

X = pddata.iloc[:,1:12]
X=np.array(X)




from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42,max_depth=None)
rf.fit(X, y)

dictionarypro = {'MacBook Pro':{'price':'1803.6','image':'static/images/macbook_pro.jpg'},
                 'Aspire 3':{'price':'400','image':'static/images/Aspire_3.png'},
                 'Macbook Air':{'price':'1158.7','image':'static/images/macbook_air.jpg'},
                 'Inspiron 3567':{'price':'639','image':'static/images/inspiron_3567.jpg'},
                 'Inspiron 7567':{'price':'899','image':'static/images/inspiron_7567.jpg'},
                 'Inspiron 5579':{'price':'1049','image':'static/images/inspiron_5579.jpg'},
                 'Inspiron 5379':{'price':'890','image':'static/images/inspiron_5379.jpg'},
                 'Latitude 5590':{'price':'1298','image':'static/images/latitude_5590.jpg'},
                 'Inspiron 5570':{'price':'800','image':'static/images/inspiron_5570.jpg'},
                 'Inspiron 3576':{'price':'728','image':'static/images/inspiron_3567.jpg'},
                 '250 G6':{'price':'575','image':'static/images/250_G6.jpg'},
                 'XPS 13':{'price':'979','image':'static/images/xps_13.jpeg'},
                 'Inspiron 5577':{'price':'1060.5','image':'static/images/inspiron_5577.jpg'},
                 'Inspiron 7577':{'price':'1499','image':'static/images/inspiron_7577.jpg'},
                 'Inspiron 7773':{'price':'999','image':'static/images/inspiron_7773.jpg'},
                 'ZenBook UX430UN':{'price':'1495','image':'static/images/zenbook_ux430un.png'},
                 'XPS 15':{'price':'2397','image':'static/images/XPS_15.jpg'}}


def collab_rec(name):
    mat = newdata.pivot_table(index='user_name', columns='laptop_name', values='searched')
    laptop_user_ratings = mat[name]
    similar_to_laptop = mat.corrwith(laptop_user_ratings)
#     print(laptop_user_ratings)
    corr_laptop = pd.DataFrame(similar_to_laptop, columns=['Correlation'])
    corr_laptop.dropna(inplace=True)
    corr_laptop=corr_laptop.sort_values('Correlation', ascending=False)
    ansar = np.array(corr_laptop.index)
    newansar=[]
    prolist = ['Inspiron 3567', 'Inspiron 7567','Inspiron 5579','Inspiron 5579','Latitude 5590','Inspiron 5570','Inspiron 3576','XPS 13','Inspiron 5577','Inspiron 7577','Inspiron 7773','XPS 15']
    for i in range(len(ansar)):
        if ansar[i] in prolist:
            newansar.append(ansar[i])
    newansar = newansar[0:5]
    print('Collab Recommended: ',newansar)



def content_rec(name):
#     print(data.head(20))
    
    pdframe = data[data['Product'].str.match(name)].iloc[0]
    # print(pdframe)
    pdframe['Company'] = lecompany.transform([pdframe['Company']])[0]
    pdframe['Product'] = leproduct.transform([pdframe['Product']])[0]
    pdframe['TypeName'] = letypename.transform([pdframe['TypeName']])[0]
    pdframe['Inches'] = leinches.transform([pdframe['Inches']])[0]
    pdframe['ScreenResolution'] = lescreenresolution.transform([pdframe['ScreenResolution']])[0]
    pdframe['Cpu'] = lecpu.transform([pdframe['Cpu']])[0]
    pdframe['Ram'] = leram.transform([pdframe['Ram']])[0]
    pdframe['Memory'] = lememory.transform([pdframe['Memory']])[0]
    pdframe['Gpu'] = legpu.transform([pdframe['Gpu']])[0]
    pdframe['OpSys'] = leopsys.transform([pdframe['OpSys']])[0]
    pdframe['Weight'] = leweight.transform([pdframe['Weight']])[0]
    pdframe['Price_euros'] = leprice_euros.transform([pdframe['Price_euros']])[0]

    npar = np.array(pdframe)
    index = [0,2]
    npar=np.delete(npar,index)
    
    prolist = ['Inspiron 3567', 'Inspiron 7567','Inspiron 5579','Inspiron 5579','Latitude 5590','Inspiron 5570','Inspiron 3576','XPS 13','Inspiron 5577','Inspiron 7577','Inspiron 7773','XPS 15']
    predictions = rf.predict_proba(npar.reshape(1,-1))
    proarray = []
    lll=[]
    for i in range(len(predictions[0])):
        proarray.append(i)
    ansd={}
    predictions, proarray = zip(*sorted(zip(predictions[0], proarray),reverse = True))
    # ans=[]
    j=0
    for i in range(1,len(proarray)):
        if j>3:
            break
        if (leproduct.inverse_transform([proarray[i]])[0] in prolist):
            j=j+1
            # ans.append(leproduct.inverse_transform([proarray[i]])[0])
            imagelink=dictionarypro[leproduct.inverse_transform([proarray[i]])[0]]['image']
            price=dictionarypro[leproduct.inverse_transform([proarray[i]])[0]]['price']
            similarity=predictions[i]*1000
            prev=name
            lll.append(leproduct.inverse_transform([proarray[i]])[0])
            lll.append(price)
            lll.append(imagelink)
            lll.append(similarity)
            lll.append(prev)

            ansd[leproduct.inverse_transform([proarray[i]])[0]]=[price,imagelink,similarity,prev]
    # print(lll)
    return lll



def get_recommendations():  
    history = pd.read_csv("pref.csv",header = None)
    filtered_df = history[history[0].notnull()]
    history = filtered_df[0]
    # historynp = history[np.isfinite(history[0])]
    # history.dropna(how='any')
#     print(history.head(20))

    prods = ['MacBook Pro','Aspire 3','Macbook Air','Inspiron 3567','Inspiron 7567','Inspiron 5579','Inspiron 5379','Latitude 5590','Inspiron 5570','Inspiron 3576','250 G6','XPS 13','Inspiron 5577','Inspiron 7577','Inspiron 7773','ZenBook UX430UN','XPS 15']
    
    # countarr = []
    # for i in range(len(prods)):
    #     countarr.append(0)
    # temp=[]
    # for j in range(len(prods)):
    #     lower = prods[j].lower()
    #     for i in history:
    #         if lower in i.lower():
    #             countarr[j]+=1
    #             temp.append(prods[j])
    temp=[]
    newtemp=[]
    for i in history:
        for j in prods:
            if(j.lower() in i.lower()):
                temp.append(j)
    print(temp)
    for i in temp:
        if i in newtemp:
            continue
        else:
            newtemp.append(i)
    # temp=list(set(temp))
    print(newtemp)
    temp=newtemp[0:3]
    topnames=temp
#                 temp.append(prods[j])
#     extractedcount, extractednames = zip(*sorted(zip(countarr, prods),reverse = True))
#     topnames = list(extractednames[0:3])
# #     topnames = list(temp[0:3])
#     topcount = list(extractedcount[0:3])
#     for i in range(len(topnames)):
#         if(topcount[i]==0):
#             topnames.remove(topnames[i])
#             topcount.remove(topcount[i])
    # print(topnames)
    retlist ={}
    a=[]
    for i in topnames:
        # print('\n Because you searched: ',i)
        b=content_rec(i)
        a=a+b
        # retlist = {**retlist,**a}
        # retlist.update(a)
        # print(retlist)
        # for i in a:
        #     retlist.append(i)
    print(a)
    return a
        # collab_rec(i)


# get_recommendations()



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    retlist = get_recommendations()
    # print(retlist)
    return jsonify(retlist)


if __name__ == "__main__":
    app.run()
