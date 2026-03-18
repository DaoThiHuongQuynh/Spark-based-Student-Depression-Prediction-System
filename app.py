import sys
import os
import json
import math
import traceback
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Load giao diện
        try:
            if not os.path.exists("app.ui"):
                raise FileNotFoundError("Không tìm thấy file 'app.ui' cùng thư mục với app.py")
            uic.loadUi("app.ui", self)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Lỗi UI", f"Lỗi load giao diện:\n{e}")
            sys.exit(1)

        # --- LOAD THAM SỐ TỪ FILE JSON ---
        self.model_data = None
        json_path = "model_params.json"

        # Kiểm tra xem file JSON đã được tạo chưa (do file train_and_export.py tạo ra)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    self.model_data = json.load(f)
                print("✅ Đã load tham số mô hình thành công!")
                # In thử ra để kiểm tra
                print(f"   Intercept: {self.model_data['intercept']:.4f}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Lỗi", f"File JSON bị lỗi: {e}")
        else:
            QtWidgets.QMessageBox.critical(self, "Lỗi",
                                           "Chưa tìm thấy file 'model_params.json'!\n\nVui lòng chạy file 'train_and_export.py' trước để tạo file này.")

        self.pushButton.clicked.connect(self.on_button_click)

        # Cài đặt phím Tab để chuyển ô
        try:
            widgets = [self.textEdit, self.textEdit_3, self.textEdit_4, self.textEdit_5, self.textEdit_7]
            for w in widgets:
                w.installEventFilter(self)
        except:
            pass

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Tab:
            self.focusNextChild()
            return True
        return super().eventFilter(obj, event)

    def on_button_click(self):
        if not self.model_data:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa có mô hình (thiếu file JSON).")
            return

        try:
            # 1. Lấy dữ liệu Input từ giao diện
            age = float(self.textEdit.toPlainText())
            cgpa = float(self.textEdit_3.toPlainText())
            ap = float(self.textEdit_4.toPlainText())
            ss = float(self.textEdit_5.toPlainText())
            fs = float(self.textEdit_7.toPlainText())

            gender = 1.0 if self.comboBox_2.currentText() == "Male" else 0.0
            fh = 1.0 if self.comboBox_3.currentText() == "Yes" else 0.0

            sd_map = {"Less than 5 hours": 0.0, "5-6 hours": 1.0, "7-8 hours": 2.0, "More than 8 hours": 3.0}
            sd = sd_map.get(self.comboBox.currentText(), 1.0)

            # Tạo danh sách các đặc trưng (Đúng thứ tự lúc train)
            # ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Financial Stress", "Sleep Duration", "Gender", "Family History"]
            raw_features = [age, cgpa, ap, ss, fs, sd, gender, fh]

            # 2. TÍNH TOÁN DỰ BÁO (Bằng công thức toán học thuần túy)
            means = self.model_data["means"]
            stds = self.model_data["stds"]
            coeffs = self.model_data["coefficients"]
            intercept = self.model_data["intercept"]

            # Bước A: Chuẩn hóa (Standard Scaler) -> Công thức: z = (x - mean) / std
            scaled_features = []
            for i in range(len(raw_features)):
                if stds[i] == 0:  # Tránh chia cho 0
                    z = 0.0
                else:
                    z = (raw_features[i] - means[i]) / stds[i]
                scaled_features.append(z)

            # Bước B: Nhân hệ số (Logistic Regression) -> Công thức: logit = intercept + sum(feat * coeff)
            logit = intercept
            for i in range(len(scaled_features)):
                logit += scaled_features[i] * coeffs[i]

            # Bước C: Hàm Sigmoid (Tính xác suất) -> Công thức: P = 1 / (1 + e^-logit)
            probability = 1.0 / (1.0 + math.exp(-logit))

            # 3. HIỂN THỊ KẾT QUẢ
            status = "CÓ NGUY CƠ" if probability >= 0.5 else "BÌNH THƯỜNG"

            # Đổi màu chữ trên giao diện
            color = "red" if probability >= 0.5 else "green"
            self.label_8.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14pt;")
            self.label_8.setText(f"{status}\n({probability:.2%})")

            QtWidgets.QMessageBox.information(self, "Kết Quả Dự Báo",
                                              f"Tình trạng: {status}\nXác suất: {probability:.2%}")

        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Lỗi Nhập Liệu", "Vui lòng nhập số hợp lệ vào các ô tuổi, điểm...!")
        except Exception as e:
            print(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Lỗi Tính Toán", f"Chi tiết lỗi:\n{str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())