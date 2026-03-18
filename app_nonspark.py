# 1_NonSpark/app.py
import sys
import os
import joblib
import pandas as pd
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt


class AppNonSpark(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("app.ui", self)
        self.setWindowTitle("App Dự Báo (Phiên bản Non-Spark)")

        # Load Model Sklearn
        if not os.path.exists("model_sklearn.pkl"):
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Chưa chạy file train.py!")
            sys.exit()

        self.model = joblib.load("model_sklearn.pkl")
        self.scaler = joblib.load("scaler_sklearn.pkl")

        self.pushButton.clicked.connect(self.on_button_click)

        # Sự kiện Tab
        for w in [self.textEdit, self.textEdit_3, self.textEdit_4, self.textEdit_5, self.textEdit_7]:
            w.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Tab:
            self.focusNextChild()
            return True
        return super().eventFilter(obj, event)

    def on_button_click(self):
        try:
            # Lấy dữ liệu
            age = float(self.textEdit.toPlainText())
            cgpa = float(self.textEdit_3.toPlainText())
            ap = float(self.textEdit_4.toPlainText())
            ss = float(self.textEdit_5.toPlainText())
            fs = float(self.textEdit_7.toPlainText())

            gender = 1 if self.comboBox_2.currentText() == "Male" else 0
            fh = 1 if self.comboBox_3.currentText() == "Yes" else 0

            sleep_map = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
            sd = sleep_map.get(self.comboBox.currentText(), 1)

            # Tạo DataFrame
            input_df = pd.DataFrame([[age, gender, cgpa, ap, ss, sd, fs, fh]],
                                    columns=["Age", "Gender", "CGPA", "Academic Pressure", "Study Satisfaction",
                                             "Sleep Duration", "Financial Stress", "Family History of Mental Illness"])

            # Chuẩn hóa
            cols_scale = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Financial Stress"]
            input_df[cols_scale] = self.scaler.transform(input_df[cols_scale])

            # Dự báo
            prob = self.model.predict_proba(input_df)[0][1]
            status = "CÓ NGUY CƠ" if prob > 0.5 else "BÌNH THƯỜNG"

            self.label_8.setText(f"{status}\n({prob:.2%})")

        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Vui lòng nhập số hợp lệ!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", str(e))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AppNonSpark()
    window.show()
    sys.exit(app.exec_())