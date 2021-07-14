from pyzbar import pyzbar
import cv2
from pylibdmtx import pylibdmtx
import numpy as np
import imutils
import time
from timeit import default_timer as timer
import concurrent.futures
import threading
from PyQt5 import QtCore, QtGui, QtWidgets

startTime = timer()
threads =0
z=0
i=0

#add filename path
fileName="Angle (1).jpg"

image = cv2.imread(fileName)
original = image.copy()
original2 = image.copy()

##Initiate list_of_barcodes(total of all barcodes)
list_of_barcodes=[]

def image_processing(image,scale,brightness):
    #Resize
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height),cv2.INTER_NEAREST)

    #brightness control
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvImg[..., 2] = hsvImg[..., 2] * brightness
    image = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)

    #sharpening
    sharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpening_filter)
    return image

def segmentation(image):
    #Segmentation based on ROI
    gray2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray2, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray2, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (15, 15))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=20)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc = cnts[0] if len(cnts) == 2 else cnts[1]
    return cc

def bounding_box (image):
    #Bounding box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(18, 18))
    gray3 = clahe.apply(gray)
    gradX1 = cv2.Sobel(gray3, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY1 = cv2.Sobel(gray3, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient1 = cv2.subtract(gradX1, gradY1)
    gradient1 = cv2.convertScaleAbs(gradient1)
    blurred = cv2.blur(gradient1, (20, 20))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=3)
    closed = cv2.dilate(closed, None, iterations=3)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def rotate(countours,ROI):
    #Rotate the segment based on the biggest bounding box
    recte = cv2.minAreaRect(countours)
    boxy = cv2.cv.BoxPoints(recte) if imutils.is_cv2() else cv2.boxPoints(recte)
    boxy = np.int0(boxy)
    #cv2.drawContours(ROI, [boxy], -1, (255, 55, 90), 5)
    (x2, y2, w2, h2) = cv2.boundingRect(boxy)
    center = recte[0]
    angle = recte[2]
    MM = cv2.getRotationMatrix2D(center, angle - 90, 1)
    (h, w) = ROI.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    coss = np.abs(MM[0, 0])
    sins = np.abs(MM[0, 1])
    nWW = int((h2 * sins) + (w2 * coss))
    nHH = int((h2 * coss) + (w2 * sins))
    if nWW > nHH:
        M = cv2.getRotationMatrix2D((cX, cY), angle + 90, 1.0)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated = cv2.warpAffine(ROI, M, (nW, nH))
    return rotated

#This Function is to decode Barcode and QR code
def barcode_reader(image):
    global i
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    t, bimage = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    barcodes = pyzbar.decode(bimage)
    for barcode in barcodes:
        i = i + 1;
        (x, y, w, h) = barcode.rect
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.polylines(image, [pts], True, (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        print("No.", i, "Data :", barcodeData, "Type :", barcodeType)
        # text = "{} ({})".format(matrixData, matrixType)
        # text = str(i)
        # cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        list_of_barcodes.append(barcode)


#This Function is to decode Data Matrix code
def DataMatrix_reader(DataMatrix):
    global i
    gray = cv2.cvtColor(DataMatrix, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    msg = pylibdmtx.decode(thresh)
    for datamatrix in msg:
        matrixData = datamatrix.data
        matrixData = str(matrixData)
        matrixData =matrixData.replace("\\r\\n", "|")
        matrixType = "Data Matrix"
        i = i + 1
        print("No.", i, "Type :", matrixType,"\nData :", matrixData)
        # text = "{} ({})".format(matrixData, matrixType)
        #text = str(i)
        #cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    return msg


def SegmentationProcessor(cc):
    global z
    rect = cv2.minAreaRect(cc)
    area = cv2.contourArea(cc)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    approx = cv2.approxPolyDP(cc, 0.04* cv2.arcLength(cc, True), True)
    if area > 10000:
        (x, y, w, h) = cv2.boundingRect(approx)
        #cv2.drawContours(image, [box], 0, (200, 200, 60), 3)
        z = z + 1
        ROI = original[y - 30:y + h + 30, x - 30:x + w + 30]
        segmented = ROI
        cnts1 = bounding_box(segmented)
        cc2 = imutils.grab_contours(cnts1)
        cc2 = sorted(cc2, key=cv2.contourArea, reverse=True)[0]
        # Rotate Segmented Image
        rotated = rotate(cc2, ROI)
        rotatedcopy = rotated.copy()
        # Run Image Processing on Rotated Image
        rotated_processed = image_processing(rotatedcopy, 2, 0.9)
        # Run Pyzbar Decoder
        barcode_reader(rotated_processed)
        cnts2 = bounding_box(rotated)
        cc1 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        DataMatrixLoop(cc1, rotated, z, rotatedcopy)

def DataMatrixLoop(cc1,rotated,z,rotatedcopy):
    for c1 in cc1:
        rect1 = cv2.minAreaRect(c1)
        area1 = cv2.contourArea(c1)
        box1 = cv2.cv.BoxPoints(rect1) if imutils.is_cv2() else cv2.boxPoints(rect1)
        box1 = np.int0(box1)
        # Print Bounding Box
        if area1 > 2000:
            DataMatrixProcessor(c1, rotated, box1, z, rotatedcopy)

def DataMatrixProcessor(c1,rotated,box1,z,rotatedcopy):
    global i
    approx2 = cv2.approxPolyDP(c1, 0.01 * cv2.arcLength(c1, True), True)
    (x1, y1, w1, h1) = cv2.boundingRect(approx2)
    cv2.drawContours(rotated, [box1], -3, (255, 55, 90), 5)
    #cv2.imwrite('Zegment_{}.png'.format(z), rotated)
    try:
        bars = rotatedcopy[y1 - 20:y1 + h1 + 20, x1 - 20:x1 + w1 + 20]
        barscopy = bars.copy()
        ar = float(w1) / h1
        if ar >= 0.8 and ar <= 1.2:
            # Resize
            width = int(barscopy.shape[1] * 0.8)
            height = int(barscopy.shape[0] * 0.8)
            barscopy = cv2.resize(barscopy, (width, height))
            # Run DataMatrix decoder if bounding box is Square
            upList=DataMatrix_reader(barscopy)
            for each in upList:
                ##Test
                new_list=(str(each[0]).replace("\\r\\n", "|"),"DATA MATRIX")
                #list_of_barcodes.append(each)
                ##Test
                list_of_barcodes.append(new_list)
    except:
        pass

#Main Function
def main(image):

    cc=segmentation(image)
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        executor.map(SegmentationProcessor, cc)
        threads = threading.active_count()
    return threads


#print("Test",len(barcode_reader(image)))
# print("Test",len(DataMatrix_reader()))

#GUI Part
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(4095, 2016)
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.HeaderLabel = QtWidgets.QLabel(self.centralwidget)
        self.HeaderLabel.setMaximumSize(QtCore.QSize(1920, 1080))
        font = QtGui.QFont()
        font.setFamily("Traditional Arabic")
        font.setPointSize(15)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.HeaderLabel.setFont(font)
        self.HeaderLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.HeaderLabel.setMouseTracking(True)
        self.HeaderLabel.setTabletTracking(False)
        self.HeaderLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.HeaderLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.HeaderLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.HeaderLabel.setTextFormat(QtCore.Qt.PlainText)
        self.HeaderLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.HeaderLabel.setWordWrap(False)
        self.HeaderLabel.setObjectName("HeaderLabel")
        self.verticalLayout.addWidget(self.HeaderLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imgLbl = QtWidgets.QLabel(self.centralwidget)
        self.imgLbl.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgLbl.sizePolicy().hasHeightForWidth())
        self.imgLbl.setSizePolicy(sizePolicy)
        self.imgLbl.setMaximumSize(QtCore.QSize(int(new_width), int(new_height)))
        font = QtGui.QFont()
        font.setUnderline(False)
        self.imgLbl.setFont(font)
        self.imgLbl.setAutoFillBackground(False)
        self.imgLbl.setText("")
        self.imgLbl.setPixmap(QtGui.QPixmap(fileName))
        self.imgLbl.setScaledContents(True)
        self.imgLbl.setWordWrap(False)
        self.imgLbl.setObjectName("imgLbl")
        self.horizontalLayout.addWidget(self.imgLbl)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setRowCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Courier")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Courier")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(50)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setCascadingSectionResizes(False)
        self.horizontalLayout.addWidget(self.tableWidget)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.loadData()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def loadData(self):
        row = 0
        self.tableWidget.setRowCount(i)
        for row in range(len(list_of_barcodes)):
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(str(list_of_barcodes[row][0])[2:-1]))
            self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(str(list_of_barcodes[row][1])))
            row = row + 1

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Output"))
        self.HeaderLabel.setText(_translate("MainWindow", "Barcode Decoder"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Decoded Output"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Barcode Type"))


if __name__ == '__main__':
    ##Preprocessing
    # Check for image size
    height, width, channel = image.shape
    if (width or image) > 1500:
        new_width = 880
        new_height = (880 * height) / width
    else:
        new_width = width
        new_height = height
    thread=main(image)

    endTime = timer()
    totalTime = endTime - startTime
    print("Total Program Time:", totalTime)


    #GUI part
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())





