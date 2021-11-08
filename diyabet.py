import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

dosya = "C:/Users/Shanty/Desktop/diyabet1.csv"
veri = loadtxt(dosya, delimiter=',')

X = veri[:,0:8]
Y = veri[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(X,Y)
model.fit(X,Y,validation_split=0.33,epochs=150,batch_size=10)
print("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_accuracy"]))




plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("model başarımları")
plt.ylabel("başarım")
plt.xlabel("epok sayisi")
plt.legend(["egitim","test"],loc="upper left")
plt.show()



plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("model kayıpları")
plt.ylabel("kayıp")
plt.xlabel("epok sayisi")
plt.legend(["egitim","test"],loc="upper left")
plt.show()

log_reg_grid = {'C': np.logspace(-4,4,30),
"solver":["liblinear"]}
#setup  the gird cv
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                           verbose=True)
#fit grid search cv
gs_log_reg.fit(X,Y)
score = gs_log_reg.score(X,Y)
print(score*100)

y_preds = gs_log_reg.predict(X)

plot_roc_curve(gs_log_reg,X,Y)