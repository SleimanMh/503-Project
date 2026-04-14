# ── API Gateway + Dashboard (Container 1) ──
FROM python:3.12-slim
WORKDIR /app

# Install gateway dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dashboard dependencies
COPY dashboard/requirements.txt dashboard-requirements.txt
RUN pip install --no-cache-dir -r dashboard-requirements.txt

COPY app/ app/
COPY dashboard/ dashboard/
COPY start.sh .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
