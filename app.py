from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النموذج المدرب والمقياس
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

# إعداد نقطة النهاية للتوقع
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات المدخلة من العميل
        data = request.get_json()

        # التأكد من أن البيانات المدخلة تحتوي على 4 ميزات
        if len(data['features']) != 4:
            return jsonify({'error': 'يجب أن تحتوي البيانات المدخلة على 4 ميزات'}), 400

        # تحويل البيانات المدخلة إلى مصفوفة
        input_data = np.array([data['features']])

        # تطبيع البيانات المدخلة
        scaled_data = scaler.transform(input_data)

        # إجراء التوقع باستخدام النموذج
        prediction = model.predict(scaled_data)

        # إرسال النتيجة للمستخدم
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
