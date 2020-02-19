#!/usr/bin/python
# -*- coding: utf-8 -*-
import urllib3, requests, json
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from flask import Flask, request, render_template,send_file, make_response, send_from_directory,Response
import uuid
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import wave
from ibm_botocore.client import Config
import ibm_boto3

VISUAL_RECOGNITION_API_VERSION = '2018-03-19'
app = Flask(__name__, static_url_path='')

# COS管理画面から認証情報をコピペして下さい。
cos_credentials = {
    'IAM_SERVICE_ID': 'iam-ServiceId-bf62c3f4-292b-4d87-bb47-1084ea83dda2',
    'IBM_API_KEY_ID': 'S_4N98v8mXkJ872fIu_PujnxoiM2MqcJBBi9HO0di2Io',
    'ENDPOINT': 'https://s3.ap-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.au-syd.bluemix.net/oidc/token',
    'BUCKET': 'hokkaidouken3-donotdelete-pr-l6yokxocpilrbi'
}

# COSオブジェクトの生成
cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=cos_credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=cos_credentials['ENDPOINT'])

# Watsonを使用するための認証
if os.environ.get('BINDING'):
      credentials = json.loads(os.environ.get('BINDING'))
      authenticator = IAMAuthenticator(credentials['apikey'])
      visual_recognition = VisualRecognitionV3(
           version=VISUAL_RECOGNITION_API_VERSION,
           authenticator=authenticator
      )
      visual_recognition.set_service_url('https://api.us-south.visual-recognition.watson.cloud.ibm.com')

@app.route('/')
def index():
    return '<h1>Hello World!</h1>'

# spectrogram
@app.route('/spectrogram', methods=['POST'])
def spectrogram():
    
     #inputデータ読み取り
     u4 = str(uuid.uuid4())
     w = wave.Wave_write("static/" + u4 + ".wav")
     w.setnchannels(1)
     w.setsampwidth(2)
     w.setframerate(44100)
     w.writeframes(request.data)
     w.close()
     y, sr = librosa.load("static/" + u4 + ".wav") #波形情報とサンプリングレートを出力
     #y, sr = librosa.load('./static/normal1.wav') #波形情報とサンプリングレートを出力

     ##### スペクトログラムを表示する #####
     # フレーム長
     fft_size = 512                 
     # フレームシフト長 
     hop_length = int(fft_size / 4)  

     # 短時間フーリエ変換実行
     amplitude = np.abs(librosa.core.stft(y, n_fft=fft_size, hop_length=hop_length))

     # 振幅をデシベル単位に変換
     log_power = librosa.core.amplitude_to_db(amplitude)

     # グラフ表示
     librosa.display.specshow(log_power, sr=sr, hop_length=hop_length, cmap='magma')
     plt.savefig("static/" + u4 + ".png")
     plt.close()
     
     # ファイルをs3にアップロード
     bucket = 'hokkaidouken3-donotdelete-pr-l6yokxocpilrbi'
     cos.upload_file(Filename="static/" + u4 + ".wav", Bucket=bucket, Key= "test/" + u4 + ".wav" )
     cos.upload_file(Filename="static/" + u4 + ".png", Bucket=bucket, Key= "test/" + u4 + ".png" )
     
     #Watson 
     with open("static/" + u4 + ".png", 'rb') as images_file:
             classes = visual_recognition.classify(
                 images_file=images_file,
                 classifier_ids=["spectrogram_579117281"]).get_result()
     return json.dumps(classes)

# melspectrogram
@app.route('/melspectrogram', methods=['POST'])
def spectrogram():
    
     #inputデータ読み取り
     u4 = str(uuid.uuid4())
     w = wave.Wave_write("static/" + u4 + ".wav")
     w.setnchannels(1)
     w.setsampwidth(2)
     w.setframerate(44100)
     w.writeframes(request.data)
     w.close()
     y, sr = librosa.load("static/" + u4 + ".wav") #波形情報とサンプリングレートを出力
     #y, sr = librosa.load('./static/normal1.wav') #波形情報とサンプリングレートを出力

     ##### スペクトログラムを表示する #####
     # フレーム長
     fft_size = 512                 
     # フレームシフト長 
     hop_length = int(fft_size / 4)  

     # 短時間フーリエ変換実行
     amplitude = np.abs(librosa.core.stft(y, n_fft=fft_size, hop_length=hop_length))

     # 振幅をデシベル単位に変換
     log_power = librosa.core.amplitude_to_db(amplitude)

     # メルスペクトログラム計算
     amplitude_2 = amplitude**2
     log_stft = librosa.power_to_db(amplitude_2)
     melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=64)

     # グラフ表示
     librosa.display.specshow(melsp, sr=sr, hop_length=hop_length, cmap='magma')
     plt.savefig("static/" + u4 + ".png")
     plt.close()
     
     # ファイルをs3にアップロード
     bucket = 'hokkaidouken3-donotdelete-pr-l6yokxocpilrbi'
     cos.upload_file(Filename="static/" + u4 + ".wav", Bucket=bucket, Key= "test/" + u4 + ".wav" )
     cos.upload_file(Filename="static/" + u4 + ".png", Bucket=bucket, Key= "test/" + u4 + ".png" )
     
     #Watson 
     with open("static/" + u4 + ".png", 'rb') as images_file:
             classes = visual_recognition.classify(
                 images_file=images_file,
                 classifier_ids=["spectrogram_579117281"]).get_result()
     return json.dumps(classes)

# spectrogram
@app.route('/spectrogram_get')
def spectrogram_get():
    
     #inputデータ読み取り
     #y, sr = librosa.load(request.get_data()) #波形情報とサンプリングレートを出力
     y, sr = librosa.load('./static/normal1.wav') #波形情報とサンプリングレートを出力

     ##### スペクトログラムを表示する #####
     # フレーム長
     fft_size = 512                 
     # フレームシフト長 
     hop_length = int(fft_size / 4)  

     # 短時間フーリエ変換実行
     amplitude = np.abs(librosa.core.stft(y, n_fft=fft_size, hop_length=hop_length))

     # 振幅をデシベル単位に変換
     log_power = librosa.core.amplitude_to_db(amplitude)

     # グラフ表示
     librosa.display.specshow(log_power, sr=sr, hop_length=hop_length, cmap='magma')
     u4 = str(uuid.uuid4())
     plt.savefig("static/" + u4 + ".png")
     plt.close()
     
     #Watson 
     with open("static/" + u4 + ".png", 'rb') as images_file:
             classes = visual_recognition.classify(
                 images_file=images_file,
                 classifier_ids=["spectrogram_579117281"]).get_result()
     return json.dumps(classes)

@app.route('/favicon.ico')
def favicon():
   return ""

port = os.getenv('VCAP_APP_PORT', '5000')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port), debug=True)