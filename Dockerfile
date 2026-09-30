# The DGX-kit dashboard, as the image the installer's systemd unit runs (dgx-kit:latest).
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY dgxkit ./dgxkit
COPY tools ./tools
COPY images ./images
# Editable, so the code finds tools/ and images/ next to it, where the benchmark and the local engine builds expect them.
RUN pip install --no-cache-dir -e .
COPY --from=web /web/dist ./web/dist
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "dgxkit.app"]
