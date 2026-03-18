# 1_NonSpark/train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

print("⏳ Đang huấn luyện theo cách truyền thống (Pandas + Sklearn)...")

# 1. ĐỌC DỮ LIỆU
try:
    # Thử đọc dấu phẩy, không được thì chấm phẩy
    df = pd.read_csv('Student Depression Dataset.csv', sep=',')
    if len(df.columns) <= 1:
        df = pd.read_csv('Student Depression Dataset.csv', sep=';')
except:
    print("❌ Lỗi: Không thấy file csv!")
    exit()

# 2. XỬ LÝ (Pandas làm rất gọn)
features = ["Age", "Gender", "CGPA", "Academic Pressure", "Study Satisfaction", "Sleep Duration", "Financial Stress", "Family History of Mental Illness"]
target = "Depression"

X = df[features].copy()
y = df[target]

# Map dữ liệu
X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
X["Family History of Mental Illness"] = X["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
sleep_map = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
X["Sleep Duration"] = X["Sleep Duration"].map(sleep_map)

# Điền dữ liệu thiếu
X.fillna(X.mean(numeric_only=True), inplace=True)

# 3. CHUẨN HÓA & TRAIN
scaler = StandardScaler()
cols_to_scale = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction", "Financial Stress"]
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# 4. LƯU MÔ HÌNH (.pkl là đặc trưng của Sklearn)
joblib.dump(model, "model_sklearn.pkl")
joblib.dump(scaler, "scaler_sklearn.pkl")

print("✅ Đã tạo xong 2 file: model_sklearn.pkl và scaler_sklearn.pkl")