# import IPython
from pyexpat import model
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import librosa
import pickle
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score as adr
import pandas as pd
from sklearn.svm import SVC

class SpeechEmotionRecognition:
  def __init__(self,filename,n_mfcc):
    self.filename=filename
    self.n_mfcc=n_mfcc

  def ExtractFeatures(self):
    self.y, self.samplingrate = librosa.load(self.filename, duration=3, offset=0.5)
    self.mfcc = np.mean(librosa.feature.mfcc(y=self.y, sr=self.samplingrate, n_mfcc=self.n_mfcc).T, axis=0)
    self.mfcc=np.array(self.mfcc)

Dataset=pd.read_csv('CSV Data\Data_Aug_noise_7 - DataAugmented_N7.csv')
X= Dataset.drop(labels=['Emotion'],axis=1)
Y=Dataset.iloc[:,128]
X=X.to_numpy()
Y=Y.to_numpy()
model = SVC(gamma='auto',max_iter=80,kernel='rbf',C=4)
model.fit(X,Y)

filename='03-01-01-01-01-01-01.wav'
n_mfcc=128
obj=SpeechEmotionRecognition(filename,n_mfcc)
obj.ExtractFeatures()

z=obj.mfcc
z.shape
ypr=model.predict([z])
print(ypr)
pickle.dump(model, open('SVC.pkl','wb'))



