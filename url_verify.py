import pandas as pd
from rsa import verify
from url_feature_extractor import featureExtraction
import joblib
import xgboost as xgb
import numpy as np

import dill

def url_verify(url):
    f_list = []
    features = featureExtraction(url)
    f_list.append(features)
    feature_names = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
    features = pd.DataFrame(f_list, columns= feature_names)
    features  = features.drop(['Domain'],axis=1)
    booster  = xgb.XGBClassifier()
    booster.load_model('models/XGBoostClassifier_model.json')
    # loaded_model = joblib.load(open("models/XGBoostClassifier_model.json", "rb"))
    # result = loaded_model.predict(features.values)
    # features = np.array(features).reshape(1,-1)
    # features = xgb.DMatrix(features.values)
    # /np.array(features)).reshape(1,-1)
    # print(features)
    # 1- phishing 0-legitimate
    with open("models/lime_explainer", 'rb') as f: explainer = dill.load(f)
    lime_res = explainer.explain_instance(features.values[0], booster.predict_proba,num_features=17)
    print(type(lime_res.as_list()))
    print(type(lime_res.as_html()))
    print(type(booster.predict(features)[0]))
    print(type(booster.predict_proba(features)[0]))
    
    result = {}
    result['prediction'] = int(booster.predict(features)[0])
    result['probability'] = {'legitimate':float(booster.predict_proba(features)[0][0]),'phishing':float(booster.predict_proba(features)[0][1])}
    # result['probability'][0] = booster.predict_proba(features)[0]
    # result['probability'][1] = booster.predict_proba(features)[1]
    result['lime_explain_list'] = lime_res.as_list()
    result['lime_explain_html'] = lime_res.as_html()
    return result


if __name__ == '__main__':
    url_verify('http://446bdf227fc4.ngrok.io/xxxbank')