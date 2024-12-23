from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# تحميل البيانات المدربة والمقياس
X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')
y_test = pd.read_pickle('y_test.pkl')
scaler = pd.read_pickle('scaler.pkl')

# إنشاء نموذج SVM
model = SVC(kernel='linear', random_state=42)

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ بالنتائج باستخدام مجموعة الاختبار
y_pred = model.predict(X_test)

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# حفظ النموذج والمقياس
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
