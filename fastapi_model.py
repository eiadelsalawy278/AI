from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# تحميل النموذج المدرب
model = torch.load('advanced_iris_model.pth')
model.eval()

# إعداد مقياس لتطبيع البيانات
scaler = StandardScaler()

# إنشاء نموذج بيانات
class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# نقطة النهاية للتنبؤ
@app.post("/predict")
def predict(features: Features):
    try:
        # تحويل البيانات المدخلة إلى مصفوفة
        input_data = np.array([[features.feature1, features.feature2, features.feature3, features.feature4]])

        # تطبيع البيانات
        scaled_data = scaler.fit_transform(input_data)

        # تحويل البيانات إلى Tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # إجراء التنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)

        return {"prediction": int(predicted_class)}

    except Exception as e:
        return {"error": str(e)}

# تشغيل التطبيق
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
