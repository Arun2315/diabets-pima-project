'''
   1.no of time pregananet
   2.plasma glucose concentration a 2hours in oral glucose tolerance test
   3.diastolic blood pressure (mm hg)
   4.triceps skin food thickness (mm)
   5.2=hours serum insulin (mu U/ml)
   6.body mass index (weight in kg/(height in m)^2)
   7.diabetes pedigree function
   8.age (years)
   9.class variable(0 or 1)

'''

from numpy import loadtxt         ##data to load use numpy to excel for another file or application
from keras.models import Sequential   ##ser means order for input and hiden layer
from keras.layers import Dense
from keras.models import model_from_json
from tensorflow.keras.models import save_model

dataset = loadtxt('pima-indians-diabetes.csv' , delimiter=',')

x=dataset[:,0:8]
y=dataset[:,8]

##print("input",x)
##print("output",y)

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model train
model.fit(x, y, epochs=10, batch_size=10)

#evalution
_, accuracy = model.evaluate(x, y)
print('Accuracy : %.2f' % (accuracy*100))

#model save
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
save_model(model, "model.h5")
print("Saved model to disk")
