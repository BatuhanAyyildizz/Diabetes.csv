# Base image
FROM python:3.12


# Çalışma dizinini ayarlayın
WORKDIR /app

# Gereksinimlerinizi kopyalayın
COPY requirements.txt .

# Bağımlılıkları yükleyin
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarınızı kopyalayın
COPY . .

# Streamlit uygulamanızı çalıştırın
CMD ["streamlit", "run", "app.py"]
