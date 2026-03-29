# ── Monitoring Container (Prometheus + Grafana + MLflow) ──
# Single container running all three monitoring services via supervisord
FROM python:3.12-slim

# ── Install system packages ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor wget tar adduser libfontconfig1 musl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Prometheus ──
ARG PROMETHEUS_VERSION=2.51.2
RUN wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz" \
    && tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz" \
    && mv "prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" /usr/local/bin/ \
    && mv "prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool" /usr/local/bin/ \
    && mkdir -p /etc/prometheus /prometheus \
    && rm -rf "prometheus-${PROMETHEUS_VERSION}.linux-amd64"*

# ── Install Grafana ──
ARG GRAFANA_VERSION=10.4.1
RUN wget -q "https://dl.grafana.com/oss/release/grafana_${GRAFANA_VERSION}_amd64.deb" \
    && dpkg -i "grafana_${GRAFANA_VERSION}_amd64.deb" || true \
    && apt-get install -f -y --no-install-recommends \
    && rm -f "grafana_${GRAFANA_VERSION}_amd64.deb"

# ── Install MLflow ──
RUN pip install --no-cache-dir mlflow>=2.9.0

# ── Create data directories ──
RUN mkdir -p /mlflow/artifacts /prometheus /var/lib/grafana

# ── Copy Prometheus config ──
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# ── Copy Grafana provisioning & dashboards ──
COPY grafana/provisioning/ /etc/grafana/provisioning/
COPY grafana/dashboards/ /var/lib/grafana/dashboards/

# ── Supervisord config ──
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Prometheus on 9090 — Grafana on 3000 — MLflow on 5000
EXPOSE 9090 3000 5000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
