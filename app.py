from flask import Flask,request,jsonify,render_template
# from flask_mail import Mail,Message
import pickle
import numpy as np


app=Flask(__name__)

with open("model.pkl",'rb') as f:
    clf=pickle.load(f)


# app.config.update(
#     MAIL_SERVER ='smtp.gmail.com',
#     MAIL_PORT = 465,
#     MAIL_USE_TLS = False,
#     MAIL_USE_SSL= True,
#     MAIL_USERNAME ="aadityapritam00@gmail.com",
#     MAIL_PASSWORD ="*********"
# )
# mail=Mail(app)

@app.route('/precaution')
def precaution():
    return render_template('precaution.html')

@app.route('/', methods=["GET","POST"])
def take_input():
    if request.method =="POST":
        myDic= request.form
        name=str(myDic['fullname'])
        cough=int(myDic['cough'])
        runny_nose=int(myDic['runny_nose'])
        fever=int(myDic['fever'])
        chest=int(myDic['chest'])
        breath=int(myDic['breathing'])
        headache=int(myDic['headache'])
        pneumonia=int(myDic['pneumonia'])
        age=int(myDic['age'])
        gender=int(myDic['gender'])
        travell=int(myDic['travel_15days'])


        # print(request.form)
    #code for predic on given input 
        var=[[cough,runny_nose,fever,chest,breath,headache,pneumonia,age,gender,travell]]
        ans1=clf.predict(var)
        prob=clf.predict_proba(var)[0][1]*100
        l=[]
        if(ans1==1):
            ans="Positive, Please contact nearest health centre"
            if(cough==1):
                l.append('cough')
            if(runny_nose==1):
                l.append('runny_nose')
            if(fever>=98):
                fv="fever:"+ str(fever)
                l.append(fv)
            if(chest>=0):
                l.append("chest tightness")
            if(breath>=0):
                l.append("breathing Problem")
            if(headache>-1):
                l.append("headche")
            if(pneumonia>0):
                l.append("Pneumonia")
            if(travell>0):
                l.append("You travelled in last 15 days , so there may be more chances matching with corona-patients")
            
            prob=prob+30
            
            # msg = Message(subject="Hi there, this person -"+name,
            #     sender="covid-19helpline@ac.in",
            #     recipients=["aadityapritam00@gmail.com"],
            #     body="Needs attention as they may have possibility or symptoms matches with Corona Patients")
            # mail.send(msg)
            
        
        else:
            ans="Negative"
            l.append("None")

       
        return render_template('result.html',ans=ans,prob=round(prob),lst=l)
        
    

        # print( "answer is" + str(ans))
        # print("probability is"+ str(prob))
    return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)
