# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import time
import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = 'xxx'
appkey = 'xxx'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'

endpoint = 'xxx'
path = 'xxx'
url = endpoint + path

query = 'Hello World! This is 1st paragraph.\nThis is 2nd paragraph.'

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

salt = random.randint(32768, 65536)
sign = make_md5(appid + query + str(salt) + appkey)

# # Build request
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
# payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

# # Send request
# r = requests.post(url, params=payload, headers=headers)
# result = r.json()

# # Show response
# print(json.dumps(result, indent=4, ensure_ascii=False))

last_run_time = None
min_interval = 3.0  # 最小调用间隔，单位秒

def translate_en_to_zh(text: str) -> str:
    """
    使用百度翻译API将文本从英文翻译为简体中文。
    
    Args:
        text: 需要翻译的英文文本
        
    Returns:
        str: 翻译后的简体中文文本
    """

    global last_run_time
    current_time = time.time()
    # 限制每次调用间隔，避免频繁请求
    if last_run_time is not None:
        elapsed_time = current_time - last_run_time
        if elapsed_time < min_interval:
            time.sleep(min_interval - elapsed_time)
    last_run_time = time.time()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)
    
    payload = {
        'appid': appid,
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'salt': salt,
        'sign': sign
    }
    
    try:
        r = requests.post(url, params=payload, headers=headers, timeout=5)
        result = r.json()
    except Exception as e:
        return f'翻译失败'

    if 'trans_result' in result:
        translated_text = '\n'.join([item['dst'] for item in result['trans_result']])
        return translated_text
    else:
        return '翻译失败'
    
if __name__ == "__main__":
    sample_text = "This is a sample text to be translated."
    translated = translate_en_to_zh(sample_text)
    print("Translated Text:", translated)
