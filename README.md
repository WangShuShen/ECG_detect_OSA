# ECG_detect_OSA

------------------------------------------
This program is designed to recognize **Obstructive sleep apnea**. Using I²C connect MAX86150 with ARC EM9D board GMA303KU on board and Convolutional Neural Networks to recognize these diease.

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176604629-1e02e3cf-c78f-47bf-a87c-b511c30e8c5f.png">

* [Introduction](#introduction)
* [Hardware and Software Setup](#hardware-and-software-setup)
	* [Required Hardware](#required-hardware)
	* [Required Software](#required-software)
	* [Hardware Connection](#hardware-connection)
* [User Manual](#user-manual)
	* [Compatible Model](#compatible-model)
	* [C work on board](#c-work-on-board)
		* [Initial each device](#initial-each-device)
	* [Python work for deep learning](#python-work-for-deep-learning)
		* [Loading dataset and pre-processing](#loading-dataset-and-pre-processing)
		* [Setup model](#setup-model)
	


## Introduction
- 1 We use 3-axis accelerometer to determine user whether is go to sleep.
- 2 Using I²C to trasport ECG data to ARC EM9D board.
- 3 Using Convolutional Neural Networks model to recognize real time ECG data.
- 4 Show the result to user.

All hardware are in the picture following:

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176442424-26c242db-f6ff-4690-a84b-176652868726.png">

## Hardware and Software Setup
### Required Hardware
-  ARC EM9D board
-  MAX86150
-  OLED1306

All hardware are in the picture following:

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176450960-a8cc7ce9-5fe4-49f3-83ad-ed625e810ab7.png">

### Required Software
- Metaware or ARC GNU Toolset
- Serial port terminal, such as putty, tera-term or minicom
- VirtualBox(Ubuntu 20.04)
- Cygwin64 Terminal

### Hardware Connection
- ARC EM9D, MAX861150 and OLED1306 connected by wire.

## User Manual

### Compatible Model

1. Download [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)

1. Give the **Apnea-ECG Database** which held Computers contest with Cardiology in 2000.

1. Use **Tensorflow 2.x** and **Python 3.7 up** to training model

1. Import module to read Apnea-ECG Database

```python
import wfdb
```
### C work on board
#### Initial each device

- Open workshop/Synopsys_SDK_V22/Example_Project/Lab2_I2C_OLED to know I²C Serial Transport.

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "embARC.h"
#include "embARC_debug.h"
#include "board_config.h"
#include "arc_timer.h"
#include "hx_drv_spi_s.h"
#include "spi_slave_protocol.h"
#include "hardware_config.h"

#include "hx_drv_iic_m.h"
#include "synopsys_i2c_oled1306.h"

#include "hx_drv_uart.h"
#define uart_buf_size 100

#define USE_SS_IIC_X USE_SS_IIC_1

DEV_UART *uart0_ptr;
char uart_buf[uart_buf_size] = {0};

DEV_IIC *iic1_ptr;
int main(void)
{
    // UART 0 is already initialized with 115200bps
    printf("This is Lab2_I2C_OLED\r\n");

    uart0_ptr = hx_drv_uart_get_dev(USE_SS_UART_0);

    sprintf(uart_buf, "I2C0 Init\r\n");
    uart0_ptr->uart_write(uart_buf, strlen(uart_buf));
    board_delay_ms(10);

    iic1_ptr = hx_drv_i2cm_get_dev(USE_SS_IIC_1);
    iic1_ptr->iic_open(DEV_MASTER_MODE, IIC_SPEED_FAST);
    
    //I2C read
    hx_drv_i2cm_read_data(USE_SS_IIC_X, GMA303KU_ADDRESS, read_buf, read_len);
    //I2C write
    a = hx_drv_i2cm_write_data(USE_SS_IIC_X, 0xbc, &data_write[0], 0, &data_write[0], 2); 
    
    //code
 }
```
- Initial MAX86150 to send ECG data.


```c
void InitMax86150(void){
    InitI2C();
    uint8_t initial_data_write[8][2] = { 
                                         {0x02, 0x80}, //Interrupt Enable 1
                                         {0x03, 0x04}, //Interrupt Enable 2
                                         {0x08, 0x10}, //FIFO Configuration 
                                         {0x09, 0x09}, //FIFO Data Control Register 1
                                         {0x0a, 0x00}, //FIFO Data Control Register 2
                                         {0x3c, 0x03}, //ECG Configuration 1   //0x02   0x03
                                         {0x3e, 0x00}, //ECG Configuration 3    //0x0D  0x00
                                         {0x0d, 0x04}}; //System Control; 
    for(int i = 0; i < 8; i++){
        Max86150_WriteData(initial_data_write[i][0], initial_data_write[i][1]);
    }
    board_delay_ms(500);
        OLED_Init();

	OLED_Clear();                        
	OLED_SetCursor(0, 0);
    OLED_DisplayString("Welcome To ARC EVK");

	OLED_SetCursor(1, 0);
	for(oled_i = 0; oled_i < 128; oled_i ++)
		oledSendData(oled_i);
}
```
- Initial OLED1306 to show result.
```c
OLED_Init();

OLED_Clear();                        
OLED_SetCursor(0, 0);
OLED_DisplayString("Welcome To ARC EVK");

OLED_SetCursor(1, 0);
for(oled_i = 0; oled_i < 128; oled_i ++)
	oledSendData(oled_i);
```

- Initial GMA303KU to send 3-axis acceleration.
```c
int16_t accel_x;
int16_t accel_y;
int16_t accel_z;
int16_t accel_t;

uint8_t xg_sign;
uint8_t yg_sign;
uint8_t zg_sign;
uint8_t temp_sign;

int16_t xg_10x;
int16_t yg_10x;
int16_t zg_10x;
int16_t temp_10x;
int main(){

    //your code
     iic1_ptr = hx_drv_i2cm_get_dev(USE_SS_IIC_0);
    iic1_ptr->iic_open(DEV_MASTER_MODE, IIC_SPEED_STANDARD); 
    uint8_t chip_id = GMA303KU_Init();
    board_delay_ms(100);

    if(chip_id == 0xA3)
        sprintf(uart_buf, "Chip ID: 0x%2X | OK\r\n\n", chip_id);    //It should be 0xA3
    else 
        sprintf(uart_buf, "Chip ID: 0x%2X | Error\r\n\n", chip_id);    //It should be 0xA3
    uart0_ptr->uart_write(uart_buf, strlen(uart_buf));
    board_delay_ms(10);
    //your code
}
```
#### Project programming
-Open tflitemicro_algo.cpp, put layers we have in model.
```cpp
	static tflite::MicroMutableOpResolver<8> micro_op_resolver;
	micro_op_resolver.AddMul();
	micro_op_resolver.AddAdd();
	micro_op_resolver.AddConv2D();
	micro_op_resolver.AddFullyConnected();
	micro_op_resolver.AddReshape();
	micro_op_resolver.AddMaxPool2D();
	micro_op_resolver.AddSoftmax();
	micro_op_resolver.AddRelu();
	// micro_op_resolver.AddMaxPool2D(
```
```cpp
```
```cpp
```
```cpp
```
### Python work for deep learning
#### Loading dataset and pre-processing
- Includes module
```python
import wfdb
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization,Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

- Load testing dataset

```python
adr =1
if adr ==1:
 change_link='/content/drive/MyDrive/CNN/ECG_model/sleep_apena_detect/apnea-ecg-database-1.0.0/'
elif adr == 2:
  change_link='/content/drive/MyDrive/ecg_data/apnea-ecg-database-1.0.0/'

x_test=[]
y_test=[]

test_string=['x01','x02','x03','x04','x05','x06','x07','x08','x09','x10','x11','x12',
             'x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24',
             'x25','x26','x27','x28','x29','x30','x31','x32','x33','x34','x35']

             
test_label_count=[522,468,464,481,504,449,508,516,507,509,456,526,505,489,497,514,399,458,486,
                  512,509,481,526,428,509,519,497,494,469,510,556,537,472,474,482]

test_label=np.loadtxt(change_link+'test-dataset-annos.txt',delimiter='\t',dtype=np.str)



tmp=[]

for i in test_label:
    tmp+=list(i)


for i in range(len(tmp)):
    if tmp[i]=='N':
        tmp[i]=0
    else:
        tmp[i]=1
y_test=np.array(tmp,dtype=np.int8)

count=0
for k in test_string:
    record = wfdb.rdrecord(record_name=change_link+k,return_res=16,physical=True)
    ecg_signal = record.p_signal
    ecg_signal=np.delete(ecg_signal,np.s_[((test_label_count[count])*6000):])
    x_test=np.append(x_test,ecg_signal)
    x_test = x_test.astype(np.float32)
    count+=1
```
- Load training dataset
```python
x_train=[]
y_train=[]
list_string=['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12',
             'a13','a14','a15','a16','a17','a18','a19','a20','b01','b02','b03','b04',
             'b05','c01','c02','c03','c04','c05','c06','c07','c08','c09','c10']

for k in list_string:

    record = wfdb.rdrecord(record_name=change_link+k,return_res=16,physical=True)
    ann = wfdb.rdann(change_link+k,'apn')

    ecg_signal_label=ann.symbol
    ecg_signal_label=np.array(ecg_signal_label)

    ecg_signal = record.p_signal


    ecg_signal=np.delete(ecg_signal,np.s_[((len(ecg_signal_label)-1)*6000):])
    x_train=np.append(x_train,ecg_signal)
    x_train = x_train.astype(np.float32)

    for i in range(len(ecg_signal_label)):
        if ecg_signal_label[i]=='N':
            ecg_signal_label[i]=0
        else:
            ecg_signal_label[i]=1
    ecg_signal_label=np.array(ecg_signal_label,dtype=np.int8)
    ecg_signal_label=np.delete(ecg_signal_label,np.s_[(len(ecg_signal_label)-1):])
    y_train=np.append(y_train,ecg_signal_label)
```
- Merge two dataset in one
```python
tmp_data=np.append(x_train,x_test,)
tmp_label=np.append(y_train,y_test)
```
- Dataset normalization
```python
tmp_data_max_abs=np.max(np.abs(tmp_data))
print(tmp_data)
tmp_data=tmp_data/tmp_data_max_abs
print(tmp_data)
```
- Split dataset
```python
tmp_data=np.reshape(tmp_data,(int(tmp_data.size/6000),6000,1,1))

x_train, x_test, y_train, y_test = train_test_split(tmp_data, tmp_label, test_size=0.1)
x_train.shape,y_train.shape ,x_test.shape , y_test.shape
```
- Flatten Label
```python
y_train = y_train.flatten()
y_test = y_test.flatten()
```
- Encoder Label
```python
num_classes=2
y_train_encoder = sklearn.preprocessing.LabelEncoder()
y_train_num = y_train_encoder.fit_transform(y_train)
y_train_wide = tensorflow.keras.utils.to_categorical(y_train_num, num_classes)
y_test_num = y_train_encoder.fit_transform(y_test)
y_test_wide = tensorflow.keras.utils.to_categorical(y_test_num, num_classes)
```
#### Setup model
- Use 2D CNN to deal with this problem
```python
model_ecg = Sequential()

model_ecg.add(BatchNormalization(input_shape=(6000,1,1)))

model_ecg.add(Conv2D(8,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(8,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Conv2D(16,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(16,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Conv2D(32,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(32,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Dropout(0.5))

model_ecg.add(BatchNormalization())
model_ecg.add(Flatten())
model_ecg.add(Dropout(0.5))

model_ecg.add(Dense(64,activation='relu'))
model_ecg.add(Dense(2,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation='softmax'))

model_ecg.summary()
```
- Choose optimizer and loss function
```python
opt = tensorflow.keras.optimizers.Adam(lr=0.001)
model_ecg.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```
- Select model's hyper parameter
```python
batch_size = 16
epochs = 300
```

- Get ready to train model
```python
best_weights_filepath_ecg = './best_weights_ecg_32layer.hdf5'
mcp_ecg = ModelCheckpoint(best_weights_filepath_ecg, monitor="val_accuracy",
                      save_best_only=True, save_weights_only=False)
history = model_ecg.fit(x_train, 
                        y_train_wide,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[mcp_ecg])
```
- Plot accuracy and loss
```python
loss = history.history['loss']
val_loss = history.history['val_loss']

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

plt.figure(figsize=(6,12))

plt.subplot(2,1,1)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Loss vs Epochs')
plt.legend()

plt.subplot(2,1,2)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(acc, 'blue', label='Training Accuracy')
plt.plot(val_acc, 'green', label='Validation Accuracy')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Accuracy vs Epochs')
plt.legend()
plt.savefig("plots_perf.svg")
plt.show()
```
- Check our confusion matrix 
```python
model_ecg.load_weights('best_weights_ecg_32layer.hdf5')
y_pred = model_ecg.predict(x_test)
predict_test=np.argmax(y_pred, axis=1)
predict_test=predict_test.reshape(predict_test.shape[0],1)
cm=confusion_matrix(y_test_num, predict_test)
cm
```
- Check test data accuracy
```python
(cm[1,1]+cm[0,0])/(cm[1,1]+cm[1,0]+cm[0,0]+cm[0,1])
```
- Save module and weight
```python
model_ecg.save_weights('model_weights.h5')
model_ecg.save('model_weights.h5')
```
- Convert 'h5' file to 'tflite' file
```python
import tensorflow as tf
model = tf.keras.models.load_model('model_weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model_ecg)
tflite_model = converter.convert()
converter.inference_input_type, converter.inference_output_type
import pathlib
generated_dir = pathlib.Path("generated/")
generated_dir.mkdir(exist_ok=True, parents=True)
converted_model_file = generated_dir/"ecg_model.tflite"
converted_model_file.write_bytes(tflite_model)  
```
- Check 'tflite' file's accuracy
```python
import tensorflow as tf
max_samples = 17233
converted_model_file="generated/ecg_model.tflite"
interpreter = tf.lite.Interpreter(model_path=str(converted_model_file))
interpreter.allocate_tensors()

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    scale, zero_point = interpreter.get_output_details()[0]['quantization']

    prediction_values = []
    
    for test_image in x_test[:max_samples]:
        # Pre-processing: add batch dimension, quantize and convert inputs to int8 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0) #.astype(np.float32)
        test_image = np.float32(test_image)
        interpreter.set_tensor(input_index, test_image)

        interpreter.invoke()

        # Find the letter with highest probability
        output = interpreter.tensor(output_index)
        result = np.argmax(output()[0])
        prediction_values.append(result)
    
    accurate_count = 0
    for index in range(len(prediction_values)):
        if prediction_values[index] == y_test[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_values)
    return accuracy * 100

print(str(evaluate_model(interpreter)) + "%")
```
- Open **Cygwin64 Terminal** press command
```c
xxd -i model.tflite > model.h
```
And you get .h file to put in ARC EM9D
