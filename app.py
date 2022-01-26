from flask import Flask, request
import re
import url_verify
import sms_text_verify
import json


app = Flask(__name__)
text_model = sms_text_verify.eng_news_zero_shot_model_pred()

@app.route('/sms_check', methods=['GET', 'POST'])
def method_name():
    data  = request.get_json()
    sms = data['sms']
    url =  re.findall(r'https?://\S+|www\.\S+', sms)
    result = dict()
    if url:
        # result['url'] = url[0]
        result['url_result'] = url_verify.url_verify(url[0])
    
    # text_model = sms_text_verify.eng_news_zero_shot_model_pred()
    result['sms_result'] = (text_model.predict(sms))
    json.dump(result, open('result.json', 'w'),separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    res = json.loads(open('result.json').read())

    print(res)
    return res

if __name__ == '__main__':
    app.run(debug=True)