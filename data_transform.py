import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('student_lifestyle_100k.csv', sep=';')

# 1. Chuẩn hóa Depression
df['Depression'] = df['Depression'].map({
    True: 1, False: 0,
    'TRUE': 1, 'FALSE': 0,
    'true': 1, 'false': 0
}).fillna(0).astype(int)
df['Age'] = df['Age'].astype(float)

# 2. Chuyển Sleep_Duration thành nhãn chuỗi
def map_sleep(hours):
    if hours < 5: return "Less than 5 hours"
    if 5 <= hours <= 6: return "5-6 hours"
    if 7 <= hours <= 8: return "7-8 hours"
    return "More than 8 hours"

df['Sleep Duration'] = df['Sleep_Duration'].apply(map_sleep)
df['CGPA'] = (df['CGPA']*2.5).round(2)

# 3. Đổi tên cột
df = df.rename(columns={'Stress_Level': 'Academic Pressure'})
df = df.rename(columns={'Study_Hours': 'Work/Study Hours'})
df = df.rename(columns={'Student_ID':'id'})
# Quy đổi Academic Pressure về thang 5 (file 100k đang là thang 10)
df['Academic Pressure'] = (df['Academic Pressure'] / 2).round()

# 4. Xử lý Study Satisfaction dựa trên Academic Pressure (Dùng np.where)
# Nếu áp lực >= 4 thì hài lòng thấp (1-2), ngược lại hài lòng cao (3-5)
df['Study Satisfaction'] = np.where(df['Academic Pressure'] >= 4,
                                    np.random.randint(1, 3, size=len(df)),
                                    np.random.randint(3, 6, size=len(df)))


df['Study Satisfaction'] = df['Study Satisfaction'].astype(float)
# 5. Xử lý Financial Stress dựa trên CGPA (Thang 4.0 ở file 100k)
# Nếu CGPA < 2.5 (tương đương < 6.5 thang 10) thì stress cao
# Cách viết chuẩn để np.where hoạt động đúng với random
raw_stress = (10 - df['CGPA']) * 0.4 + (df['Work/Study Hours'] / 5) * 0.3
df['Financial Stress'] = raw_stress + np.random.uniform(0, 1, size=len(df))
df.loc[df['Depression'] == 1, 'Financial Stress'] += 1.0

# Giới hạn trong khoảng 1.0 đến 5.0 và làm tròn 1 chữ số thập phân
df['Financial Stress'] = df['Financial Stress'].clip(1.0, 5.0).round(1)
# 6. Xử lý Dietary Habits (3 mức độ: Healthy, Unhealthy, Moderate)
# Lưu ý: Phải có ngoặc đơn () cho mỗi điều kiện so sánh
conditions = [
    # Nhóm Unhealthy: Trầm cảm HOẶC (Vận động thấp AND Áp lực cao)
    (df['Depression'] == 1) | ((df['Physical_Activity'] < 80) & (df['Academic Pressure'] >= 4)),

    # Nhóm Healthy: Vận động tốt AND Áp lực thấp AND Không trầm cảm
    (df['Physical_Activity'] > 110) & (df['Academic Pressure'] <= 2) & (df['Depression'] == 0),

    # Nhóm Moderate: Vận động mức trung bình (Từ 80 đến 110)
    (df['Physical_Activity'] >= 80) & (df['Physical_Activity'] <= 110)
]

choices = ['Unhealthy', 'Healthy', 'Moderate']

# Sử dụng np.select để gán nhãn, mặc định là Moderate nếu không rơi vào các nhóm trên
df['Dietary Habits'] = np.select(conditions, choices, default='Moderate')

# Gán Family History
df['Family History of Mental Illness'] = np.where(df['Depression'] == 1, 'Yes', 'No')
# 7. Thêm cột Suicidal Thoughts (vì file Depression có cột này và nó rất quan trọng)
df['Have you ever had suicidal thoughts ?'] = np.where(df['Depression'] == 1, 'Yes', 'No')
df['Profession'] = 'Student'
df['City'] = 'Unknown'
df['Degree'] = 'BA'
# Xuất file
df.to_csv('Student Depression Dataset 2.csv', index=False, sep=';')
print("Đã xử lý xong 100k dòng dữ liệu!")
