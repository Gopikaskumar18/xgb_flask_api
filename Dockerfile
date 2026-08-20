FROM python:3.12-slim

WORKDIR /app

# install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app + model artifacts
COPY app.py .
COPY final_xgb_model.json .
COPY label_encoders.pkl .
COPY feature_medians.pkl .

EXPOSE 8080

# production server (not Flask's dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", "app:app"]
