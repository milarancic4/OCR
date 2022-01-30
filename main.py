import pytesseract
from pytesseract import Output
import cv2, os, sys
from PIL import Image
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
import glob
import pandas as pd
import re
from Preprocessing_Image import PreprocessingImage

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# font_list = []
# font = 2
#
# for font in range(110):
#     font += 2
#     font_list.append(str(font))

pattern_list_name = [' ', 'Email', 'Name', 'Date', 'Total sum']

rotated_image = ""
r = ""

class PyShine_OCR_APP(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('pythonUI.ui', self)
        self.image = None

        self.ui.pushButton.clicked.connect(self.open)
        self.ui.pushButton_3.clicked.connect(self.write_formated_text)
        self.ui.pushButton_4.clicked.connect(self.rotate_img)
        self.ui.pushButton_2.clicked.connect(self.save)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)

        self.pattern_name = ' '
        self.comboBox.addItems(pattern_list_name)
        self.comboBox.currentIndexChanged['QString'].connect(self.update_pattern)
        self.comboBox.setCurrentIndex(pattern_list_name.index(self.pattern_name))

        # self.font_size = '20'
        # self.text = ''
        # self.comboBox.addItems(font_list)
        # self.comboBox.currentIndexChanged['QString'].connect(self.update_font_size)
        # self.comboBox.setCurrentIndex(font_list.index(self.font_size))
        #
        # self.ui.textEdit.setFontPointSize(int(self.font_size))
        # self.setAcceptDrops(True)

    # def update_font_size(self, value):
    #     self.font_size = value
    #     self.ui.textEdit.setFontPointSize(int(self.font_size))
    #     self.text = self.ui.textEdit.toPlainText()
    #     self.ui.textEdit.setText(str(self.text))
    def clear_textBox(self):
        text = self.ui.textEdit.toPlainText()
        self.ui.textEdit.setText(str(text))

    def open(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Select File')
        self.image = cv2.imread(str(self.filename[0]))
        # self.image = cv2.resize(self.image, (627, 797))
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QPixmap.fromImage(image))

    def image_to_text(self, input_img):
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 1)
        crop = Image.fromarray(gray)
        text = pytesseract.image_to_string(crop)
        # print('Text:', text)
        return text

    def image_to_data(self, input_img):
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        custom_config = r'-l eng --oem 1 --psm 4'
        d = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
        df = pd.DataFrame(d)
        df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
        text2 = ''
        for block in sorted_blocks:
            curr = df1[df1['block_num'] == block]
            sel = curr[curr.text.str.len() > 3]
            char_w = (sel.width / sel.text.str.len()).mean()
            prev_par, prev_line, prev_left = 0, 0, 0
            text = ''
            for ix, ln in curr.iterrows():
                # add new line when necessary
                if prev_par != ln['par_num']:
                    text += '\n'
                    prev_par = ln['par_num']
                    prev_line = ln['line_num']
                    prev_left = 0
                elif prev_line != ln['line_num']:
                    text += '\n'
                    prev_line = ln['line_num']
                    prev_left = 0

                added = 0  # num of spaces that should be added
                if ln['left'] / char_w > prev_left + 1:
                    added = int((ln['left']) / char_w) - prev_left
                    text += ' ' * added
                text += ln['text'] + ' '
                prev_left += len(ln['text']) + added + 1
            text += '\n'
            text2 += text
        return text2

    def write_formated_text(self):
        global r, rotated_image
        self.clear_textBox()
        if r == 'X':
            image = rotated_image
        else:
            image = cv2.imread(str(self.filename[0]))
        self.text = self.image_to_data(image)
        self.ui.textEdit.setText(str(self.text))
        r = ""
        rotated_image = ""

    def update_pattern(self, value):
        pattern_name = value
        if pattern_name == 'Email':
            pattern = "^[a-zA-Z]+[@][a-zA-Z]+[\.][a-zA-Z]+$"
            element = self.find_email(pattern)
        if pattern_name == 'Date':
            pattern = "^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$"
            element = self.find_data(pattern)
        if pattern_name == 'Total sum':
            pattern = "^[a-zA-Z]*[\s]*Total$"
            element = self.find_total_sum(pattern)
        if pattern_name == 'Name':
            pattern = "^(SHIPPING|shipping|BILL|bill)+[\s]*(TO,to)*[:]*"
            element = self.find_name(pattern)
        self.ui.textEdit.setText(str(element))

    def find_name(self, chosen_pattern):
        d = pytesseract.image_to_data(self.image, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(float(d['conf'][i])) > 60:
                if re.match(chosen_pattern, d['text'][i]):
                    found = False
                    j = i + 1
                    while j < n_boxes:
                        print(d['text'][j], d['left'][j])
                        if abs(d['left'][j] - d['left'][i]) < 10 and not found and d['text'][j] != ' ' and \
                                d['text'][j] != '' and d['conf'][j] != -1:
                            found = True
                            (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                            print(x, y, w, h)
                            self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],
                                           QImage.Format_RGB888)
                            self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                            return d['text'][j]
                        j += 1

    def find_total_sum(self, chosen_pattern):
        d = pytesseract.image_to_data(self.image, output_type=Output.DICT)
        pattern_money = "^[\$,\£]*[\d]*(,|.)[\d]*$"
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(float(d['conf'][i])) > 60:
                if re.match(chosen_pattern, d['text'][i]):
                    j = i + 1
                    print(d['text'][i])
                    if re.match(pattern_money, d['text'][j]):
                        print(d['text'][j])
                        (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                        self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],
                                       QImage.Format_RGB888)
                        self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                        print(d['text'][j])
                        return d['text'][j]

    def find_data(self, chosen_pattern):
        d = pytesseract.image_to_data(self.image, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(float(d['conf'][i])) > 60:
                if re.match(chosen_pattern, d['text'][i]):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],
                                   QImage.Format_RGB888)
                    self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                    return d['text'][i]
    def find_email(self, chosen_pattern):
        d = pytesseract.image_to_data(self.image, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(float(d['conf'][i])) > 60:
                if re.match(chosen_pattern, d['text'][i]):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],
                                   QImage.Format_RGB888)
                    self.ui.label_2.setPixmap(QPixmap.fromImage(image))
                    return d['text'][i]
    
    def rotate_img(self):
        global r, rotated_image
        r = 'X'
        a = PreprocessingImage()
        image = cv2.imread(str(self.filename[0]))
        deskewed = a.deskew(image)
        rotated_image = deskewed
        height, width, channel = deskewed.shape
        bytesPerLine = 3 * width
        qImg = QImage(deskewed.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_2.setPixmap(QtGui.QPixmap(qImg))
    
    def save(self):
        a = PreprocessingImage()
        mytext = self.textEdit.toPlainText()
        with open('test.txt', 'w') as outfile:
            outfile.write(mytext)

# www.pyshine.com
app = QtWidgets.QApplication(sys.argv)
mainWindow = PyShine_OCR_APP()
mainWindow.show()
sys.exit(app.exec_())
