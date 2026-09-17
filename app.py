from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# อนุญาตให้ React (ซึ่งรันคนละ Port) สามารถส่งข้อมูลมาหา Flask ได้
CORS(app) 

# 1. โหลดโมเดลที่เราเทรนไว้ขึ้นมาเตรียมพร้อม
model = joblib.load('rf_model.pkl')

# 2. สร้าง API Endpoint ชื่อ /predict สำหรับรับข้อมูล
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # รับข้อมูลสเปกมือถือที่ผู้ใช้กรอกผ่านหน้าเว็บ (มาในรูปแบบ JSON)
        data = request.get_json()
        
        # จัดเรียง Feature ให้ตรงกับตอนที่เราเทรนโมเดลเป๊ะๆ
        features = [
            'Storage', 
            'RAM', 
            'Screen Size (inches)', 
            'Battery Capacity (mAh)', 
            'Camera_Total_MP', 
            'Brand_Encoded'
        ]
        
        # นำข้อมูลที่รับมา สร้างเป็นตาราง (DataFrame) 1 แถว
        input_data = pd.DataFrame([data], columns=features)
        
        # ให้โมเดลทำนายราคา
        predicted_price = model.predict(input_data)[0]
        
        # ส่งผลลัพธ์กลับไปให้หน้าเว็บแสดงผล
        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2)
        })
        
    except Exception as e:
        # ถ้ามี Error (เช่น ส่งข้อมูลมาไม่ครบ) ให้แจ้งเตือนกลับไป
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์ที่พอร์ต 5000
    app.run(debug=True, port=5000)