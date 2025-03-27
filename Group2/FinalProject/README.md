# 🇻🇳 Vietnamese Product Review Sentiment Classification  

## 📝 Overview  
This project implements a **web-based application** for **Vietnamese product review sentiment analysis**, specifically focused on comments collected from the **Shopee** e-commerce platform. The system utilizes **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify reviews as **positive** or **negative**, providing valuable insights for both businesses and consumers.  

## ✨ Features  
👉 **Vietnamese text preprocessing** using **Underthesea** package  
👉 **Text vectorization** via **TF-IDF**  
👉 **Sentiment classification** using **Support Vector Machine (SVM)**  
👉 **Web-based UI** for real-time sentiment analysis  
👉 **Ngrok deployment** for remote accessibility  

## 🛠️ Technologies Used  
🔹 **Python 3.x** – Core programming language  
🔹 **Underthesea** – Vietnamese NLP toolkit for text preprocessing  
🔹 **Scikit-learn** – ML library for TF-IDF vectorization and classification  
🔹 **Flask** – Web framework for application UI  
🔹 **Ngrok** – Expose local web server to the internet  
🔹 **Playwright** – Browser automation for web scraping  

## 👄 Installation  

### ⚡ Prerequisites  
- 🐍 **Python 3.x** installed  
- 📦 **pip** (Python package manager)  

### 🚀 Setup  
1. **Clone this repository:**  
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```  
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Download and install Ngrok** (if required for external access)  

## 🔧 Usage  
1. **Run the Flask application:**  
   ```bash
   python SEG_on_web.py
   ```  
2. **Access the web interface:**  
   - Locally: `http://localhost:5000`  
   - Via Ngrok: Use your generated public URL  
3. **Enter a Vietnamese product review** in the input field  
4. **Click "Analyze"** to classify the sentiment  

## 📂 Project Structure  
```
.
├── images/                  # Team member images
├── shopee_crawler/          # Shopee comment scraping scripts
├── templates/
│   └── index.html           # Web application layout
├── best_mode_v{x}.pkl          # Different model versions with their vectorizers
├── vectorizer_v{x}.pkl         # Corresponding vectorizer
├── SEG_on_web.py            # Main Flask application
└── README.md                # Project documentation
```

## 📊 Data Collection & Processing  
🔍 **Automated web scraping** of Shopee product reviews using **Playwright**  
📉 **Data collection** across multiple product categories  
📚 **Manual review & labeling** of comments into **positive** and **negative**  
⚖️ **Dataset balancing** for equal representation of sentiment classes  

## 🎯 Model Development  
🔎 **Text Preprocessing** – Lowercasing, normalization, tokenization, stopword removal  
📈 **Feature Engineering** – TF-IDF vectorization for better text representation  
🛠️ **Model Selection** – GridSearch tuning for best performance  
🏆 **Best Model** – **SVM achieved 90% accuracy** after dataset refinement  

## 🚀 Future Development  
🔹 **Topic Extraction** – Identify key concerns in customer reviews  
🔹 **Enhanced Data Collection** – Overcome anti-bot restrictions  
🔹 **Transformer-based Models** – Explore **BERT, PhoBERT** for better semantic understanding  

## 🖼️ Screenshots  
<!-- Replace placeholder text with actual images -->  
### 📌 Application Interface  
| ✅ Panel for stochastic analysis | ✅ Panel for batch analysis | ✅ Plot |  
|:-------------------:|:-------------------:|:----------------------:|  
| ![Panel for stochastic analysis](images/dev_1.png) | ![Panel or batch analysis](images/dev_2.png) | ![Plot](images/dev_3.png) |
## 👤 Team Members  
👨‍💻 **Nguyen Van Phong (Leader)**  – FPT University, Hue  
👨‍💻 **Dao Anh Khoa** – FPT University, Dong Thap  
👨‍💻 **Huynh Anh Phuong** – FPT University, An Giang  
👨‍💻 **Tran Trung Nhan** – FPT University, Ca Mau  
👩‍💻 **Huynh Ngoc Nhu Quynh** – FPT University, Soc Trang  

---  

💡 **License:** _[not yet]_  
📩 **Contact:** [Contact](https://linktr.ee/squarerootp) 

🔥 *If you find this project useful, give it a ⭐ on GitHub!*  
