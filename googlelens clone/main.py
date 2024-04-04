from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog



import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)

df=pd.read_csv(LABEL_MAP_URL)

classifer=tf.keras.Sequential([hub.KerasLayer(
    TF_MODEL_URL, 
    input_shape=IMAGE_SHAPE +(3,),
    output_key="predictions:logits"
)])

label_map=dict(zip(df.id, df.name))


def classifyimg(RGBimg):
    RGBimg=np.array(RGBimg)/255
    RGBimg=np.reshape(RGBimg, (1,321,321,3))
    prediction=classifer.predict(RGBimg)
    return label_map[np.argmax(prediction)]


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(541, 961)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, -46, 541, 961))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 80, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 220, 181, 231))
        self.label_2.setStyleSheet("image:url(unnamed.png)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(170, 690, 201, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(".QPushButton{\n"
    
"background-color:#EA4335;\n"
"border-radius:12px;\n"
"color:black;\n"
"border: 2px solid #f44336;\n"
"}\n"
"\n"
".QPushButton.hover{\n"
"background-color: white;\n"
"color:black;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Apps"))
        MainWindow.setWindowIcon(QIcon('unnamed.png'))
        self.label.setText(_translate("MainWindow", "Google Lens"))
        self.pushButton.setText(_translate("MainWindow", "Select Image"))
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Apps"))
        MainWindow.setWindowIcon(QIcon('unnamed.png'))
        self.label.setText(_translate("MainWindow", "Google Lens"))
        self.pushButton.setText(_translate("MainWindow", "Select Image"))
        self.pushButton.clicked.connect(self.upload_img)
    def upload_img(self):
        filename=QFileDialog.getOpenFileName()
        path=filename[0]
        path=str(path)
        print(path)
        img=cv2.imread(path)
        BGRimg=cv2.resize(img, (640,480))
        RGBimg=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGBimg=cv2.resize(RGBimg, (321,321))
        result=classifyimg(RGBimg)
        print(result)
        cv2.rectangle(BGRimg, (0, 480), (640, 425),(50, 50, 255), -2)
        cv2.putText(BGRimg, 'Predicted: {}'.format(str(result)), (20,460), cv2.FONT_HERSHEY_COMPLEX, 
                   1, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("Frame",BGRimg)
        cv2.waitKey(0)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
