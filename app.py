from flask import Flask
from flask import render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model=pickle.load(open("modelXG.pkl","rb"))

@app.route('/')
def index():
    return render_template("hearthtml.html")

@app.route('/predict', methods=["POST","GET"])
def predict():
    feat=[]
    i=0
    for x in request.form.values():
        print(x)
        if(x.isdigit()):
            feat.append(int(x))
        else:
            if(i==1):
                if(x=="Male"):
                    feat.append(1)
                else:
                    feat.append(0)
            elif(i==2):
                feat.append(['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'].index(x))
            elif(i==5):
                if(x=="Yes"):
                    feat.append(1)
                else:
                    feat.append(0)
            elif(i==6):
                feat.append(['Normal', 'ST-T Wave Abnormality','Left Ventricular Hypertrophy'].index(x))
            elif(i==8):
                if(x=="Yes"):
                    feat.append(1)
                else:
                    feat.append(0)
            elif(i==10):
                feat.append(['Upsloping', 'Flat', 'Downsloping'].index(x))
            elif(i==12):
                feat.append((['Normal', 'Fixed Defect', 'Reversible Defect'].index(x))+1)
        i=+1
    final=[np.array(feat)]
    # scaler=StandardScaler()
    # scaler.fit(final)
    # final=scaler.transform(final)
    # print(final)
    predict_model=model.predict_proba(final)
    
    output="{:.1f}".format(predict_model[0][1]*100)
    if(output>=str(65)):
        return render_template("high.html",gg1="YOU ARE IN GREAT DANGER!",gg2="IMMEDIATE ATTENTION NEEDED!",pred="THE RISK OF GETTING A SEVERE HEART ATTACK IS {}%".format(output))
        # return render_template("hearthtml.html",pred="VERY HIGH RISK of HEART ATTACK \n Probability of the heart attack risk is {}%".format(output))
    elif(output<str(65) and output>=str(40)):
        return render_template("high.html",gg1="YOU ARE ON THE PATH OF GETTING HEART ATTACKS!",gg2="CONSULT A DOCTOR BEFORE IT GETS SERIOUS",pred="THE RISK OF GETTING A MILD HEART ATTACK IS {}%".format(output))
        # return render_template("hearthtml.html",pred="AVERAGE RISK of HEART ATTACK \n Probability of the heart attack risk is {}%".format(output))
    else:
        return render_template("high.html",gg1="YOU ARE SAFE FROM HEART ATTACKS!",gg2="LOW PROBABILITY OF YOU EXPERIENCING HEART ATTACKS",pred="THE RISK OF GETTING A HEART ATTACK IS {}%".format(output))
        # return render_template("hearthtml.html",pred="VERY LOW RISK of HEART ATTACK \n Probability of the heart attack risk is {}%".format(output))


if __name__ == '__main__':
    app.run(debug=True)