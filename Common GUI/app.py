from flask import Flask, render_template, request,jsonify
import os
from keras.models import load_model
import keras.utils as ima
import numpy as np
img_width, img_height = 224, 224
import librosa
import face_recognition
from torch.utils.data.dataset import Dataset
import torch
import cv2
import torchvision
from torchvision import transforms
from torch import nn
import torchvision.models as models
import time
from PIL import Image as pImage
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np

import matplotlib.pyplot as plt


im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))


train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])








app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_deepfake(file_path):
    # Replace this with your actual deepfake detection code
    # For example, you might use a pre-trained model to predict deepfake
    # and return the result.
    model = load_model('model/completed_augmented_trained_model.h5',compile=False)
    img = ima.load_img(file_path, target_size=(img_width, img_height))
    x = ima.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x=x/255.0

    result=model.predict(x)
    if(result[0]<0.50):
        result_string="It is Fake Image"
    else:
        result_string="It is real Image"

    return result_string

def detect_voicefake(filepath):
    SAMPLE_RATE=16000
    N_MELS=128
    DURATION = 5
    max_time_steps = 109

    model=load_model('model/audio_classifier.h5',compile=False)
    audio, _ =librosa.load(filepath,sr=SAMPLE_RATE,duration=DURATION)
    print(audio,"Hello")
    mel_spectro=librosa.feature.melspectrogram(y=audio,sr=SAMPLE_RATE,n_mels=N_MELS) 
    mel_spectro=librosa.power_to_db(mel_spectro,ref=np.max)
    if mel_spectro.shape[1] < max_time_steps:
        mel_spectro_new = np.pad(mel_spectro, ((0, 0), (0, max_time_steps - mel_spectro.shape[1])), mode='constant')
    else:
        mel_spectro_new = mel_spectro[:, :max_time_steps]
    
    mel_spec=np.array(mel_spectro_new)
    print(mel_spec.shape)
    mel_spec=np.expand_dims(mel_spec, axis=0)
    print(mel_spec.shape)
    result=model.predict(mel_spec)
    classes = np.argmax(result, axis=1)

    if(classes==0):
        result_string="This Voice is Spoof/Fake."
    if(classes==1):
        result_string="This is Bonafide/Real"

    return result_string


sequence_length=100

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def predict(model,img, video_file_name=""):
  fmap,logits = model(img.to('cuda'))
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)  
  return [int(prediction.item()),confidence]

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        print(video_names)
        print(sequence_length)
        print(transform)

    def __len__(self):
        print(self.video_names)
        print(len(self.video_names))
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        print(video_path)
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        #print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image






def predict_page(filename):
        video_file_name_only = filename.split('.')[0]
        vecfile=[filename]
        video_dataset = validation_dataset(vecfile, sequence_length=sequence_length,transform= train_transforms)
        
        model = Model(2).cuda()
        # model_name = os.path.join(settings.PROJECT_DIR,'models', get_accurate_model(sequence_length))
        # models_location = os.path.join(settings.PROJECT_DIR,'models')
        path_to_model = 'model/model_97_acc_100_frames_FF_data.pt'
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
    
        # Start: Displaying preprocessing images
        # print("<=== | Started Videos Splitting | ===>")
        # preprocessed_images = []
        # faces_cropped_images = []
        # cap = cv2.VideoCapture(filename)

        # frames = []
        # while(cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret==True:
        #         frames.append(frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     else:
        #         break
        # cap.release()

        # for i in range(1, sequence_length+1):
        #     frame = frames[i]
        #     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     img = pImage.fromarray(image, 'RGB')
        #     image_name = video_file_name_only+"_preprocessed_"+str(i)+'.png'
        #     # image_path = "videoImage/" + image_name
        #     image_path=os.path.join("videoImage",image_name)
        #     img.save(image_path)
        #     preprocessed_images.append(image_name)
        # print("<=== | Videos Splitting Done | ===>")
        # print("--- %s seconds ---" % (time.time() - start_time))
        # # End: Displaying preprocessing images


        # # Start: Displaying Faces Cropped Images
        # print("<=== | Started Face Cropping Each Frame | ===>")
        # padding = 40
        # faces_found = 0
        # for i in range(1, sequence_length+1):
        #     frame = frames[i]
        #     #fig, ax = plt.subplots(1,1, figsize=(5, 5))
        #     face_locations = face_recognition.face_locations(frame)
        #     if len(face_locations) == 0:
        #         continue
        #     top, right, bottom, left = face_locations[0]
        #     frame_face = frame[top-padding:bottom+padding, left-padding:right+padding]
        #     image = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

        #     img = pImage.fromarray(image, 'RGB')
        #     image_name = video_file_name_only+"_cropped_faces_"+str(i)+'.png'
        #     # image_path = "videoImage/croppedFaces/" + image_name
        #     image_path=os.path.join("videoImage/croppedFaces",image_name)
        #     img.save(image_path)
        #     faces_found = faces_found + 1
        #     faces_cropped_images.append(image_name)
        # print("<=== | Face Cropping Each Frame Done | ===>")
        # print("--- %s seconds ---" % (time.time() - start_time))

        # # No face is detected
        # if faces_found == 0:
        #     return "No face found in the given video."
        path_to_videos = [filename]
        print(path_to_videos)
        

        
        
        
        output = ""
        print("<=== | Started Predicition | ===>")
            # print(video_dataset[i],"Hellloooo")
        prediction = predict(model, video_dataset[0], video_file_name_only)
        confidence = round(prediction[1], 1)
        print("<=== |  Predicition Done | ===>")
                # print("<=== | Heat map creation started | ===>")
                # for j in range(0, sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))
        if prediction[0] == 1:
            output = "This Video is REAL"
        else:
            output = "This Video is FAKE"
        print("Prediction : " , prediction[0],"==",output ,"Confidence : " , confidence)
                
        return output
        



    

    
    

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        option = request.form['option']
        return render_template(f'{option}_detection.html')

@app.route('/upload-image', methods=['POST'])
def upload():
    if request.method == 'POST':
        option = request.form['option']
        file = request.files['file']
        # print(file,"\t",option)  
        # filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
        # file.save(filename)
        # return render_template('result.html') 

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
        else:
            return  "Invalid File"

        result = detect_deepfake(filename)

        return render_template('result.html', result=result, option=option)
    
@app.route('/upload-audio', methods=['POST'])
def uploadaudio():
    if request.method == 'POST':
        option = request.form['option']
        file = request.files['file']
        # print(file,"\t",option)  
        # filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
        # file.save(filename)
        # return render_template('result.html') 

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
        else:
            return  "Invalid File"

        result = detect_voicefake(filename)

        return render_template('result.html', result=result, option=option)
    
@app.route('/upload-video', methods=['POST'])
def uploadvideo():
    if request.method == 'POST':
        option = request.form['option']
        file = request.files['file']
        # print(file,"\t",option)  
        # filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
        # file.save(filename)
        # return render_template('result.html') 

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
        else:
            return  "Invalid File"
        
        result=predict_page(filename)
        
        return render_template('result.html', result=result, option=option)

if __name__ == '__main__':
    app.run(debug=True)
