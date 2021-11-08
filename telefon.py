import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri=pd.read_csv("C:/Users/Shanty/Desktop/telefon_fiyat_değişimi.csv")


label_encoder=LabelEncoder().fit(veri.price_range)

labels=label_encoder.transform(veri.price_range)
classes=list(label_encoder.classes_)


x=veri.drop(["price_range",'wifi','blue','dual_sim',"ram"],axis=1)
y= labels


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X= sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)




from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(16,input_dim=8,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(4,activation="softmax"))
model.summary()




model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=150)
print("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_accuracy"]))


import matplotlib.pyplot as plt


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

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
random_forest = RandomForestClassifier(max_depth=7)
random_forest.fit(X_train, y_train)
random_pred = random_forest.predict(X_test)
random_test_score = accuracy_score(y_test, random_pred)
acc_random = cross_val_score(random_forest, X_train, y_train, cv=5)
print("çapraz doğrulama skor: ",random_test_score * 100)
print(acc_random * 100)