# ML_Logistic-Regeression-Insurance-Churns
# 🛡️ Dự án Dự đoán Khách hàng Hủy Hợp đồng Bảo hiểm (Insurance Churn Prediction)

## 🌟 Tóm tắt
Dự án này tập trung vào việc xây dựng mô hình học máy để dự đoán khả năng khách hàng bảo hiểm sẽ hủy hợp đồng (*churn*). Mục tiêu là giúp công ty bảo hiểm xác định sớm các khách hàng có rủi ro cao để thực hiện các chiến lược giữ chân phù hợp.

Mô hình được xây dựng sử dụng thuật toán **Logistic Regression** và đã chứng minh hiệu suất **vô cùng mạnh mẽ**, cung cấp khả năng phân loại khách hàng tiềm năng hủy hợp đồng với độ tin cậy cao.

## 📊 Dữ liệu
Dữ liệu đầu vào được sử dụng từ tệp `randomdata.csv`. Các trường dữ liệu chính được phân tích bao gồm:
* `Customer Name`, `Customer_Address`, `Company Name`
* `Claim Reason` (Lý do yêu cầu bồi thường)
* `Data confidentiality` (Mức độ bảo mật dữ liệu)
* `Claim Amount` (Số tiền yêu cầu bồi thường)
* `Category Premium` (Mức phí bảo hiểm theo danh mục)
* `Premium/Amount Ratio` (Tỷ lệ Phí bảo hiểm/Số tiền bồi thường)
* `BMI` (Chỉ số khối cơ thể)
* `Churn` (Biến mục tiêu: **Yes** (Hủy) hoặc **No** (Không hủy))

## 🛠️ Phương pháp và Công nghệ
Dự án được triển khai bằng Python trong môi trường Jupyter Notebook và sử dụng các thư viện chính sau:
* **Modeling:** `scikit-learn` (Logistic Regression, Cross-Validation)
* **Data Analysis:** `pandas`, `numpy`
* **Visualization:** `matplotlib.pyplot`, `seaborn`, `plotly.express`
* **Others:** `pycountry-convert` (để hỗ trợ xử lý dữ liệu quốc gia/địa lý)

## ✅ Kết quả Đánh giá Mô hình
Mô hình đã được đánh giá kỹ lưỡng thông qua Cross-Validation và đạt các chỉ số ấn tượng:

| Chỉ số Đánh giá | Giá trị Trung bình | Mô tả |
| :--- | :--- | :--- |
| **Accuracy** (Độ chính xác) | **0.978** | Mô hình dự đoán đúng 97.8% các trường hợp. |
| **Macro F1 Score** | **0.977** | Chỉ số cao cho thấy mô hình cân bằng tốt, không bị thiên vị bởi lớp đa số, và phân loại tốt cho cả hai nhóm khách hàng (Hủy/Không hủy). |
| **AUC** (Area Under the Curve) | **0.9984** | Chỉ số **cực kỳ cao**, chứng tỏ khả năng phân biệt khách hàng hủy và không hủy của mô hình là **gần như hoàn hảo (99.84%)** trên mọi ngưỡng quyết định. |

## 🔑 Phân tích Chuyên sâu (Key Insight)
Mô hình bao gồm một phân tích cụ thể về ảnh hưởng của **chỉ số BMI** đối với xác suất khách hàng hủy hợp đồng:
* Đồ thị **Effect of BMI on Predicted Probability of Churn** cho thấy mối quan hệ dạng đường cong sigmoid giữa BMI và xác suất Churn. Điều này ngụ ý BMI là một trong những yếu tố quan trọng nhất ảnh hưởng đến quyết định rời bỏ của khách hàng.

## 🚀 Cách chạy Dự án
### Yêu cầu cài đặt
Để chạy notebook này, bạn cần cài đặt các thư viện Python sau:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pycountry-convert
