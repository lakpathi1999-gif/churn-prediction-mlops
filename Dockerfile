# Small, fast base image with Python pre-installed
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first (before the code) so Docker can cache
# this layer -- if only your code changes later, dependencies won't
# need to reinstall on every rebuild. Small thing, but it's a real
# "do you understand Docker" interview signal.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app.py .

# Copy the standalone UI into the image so the app can serve it at /ui
COPY ui ./ui

# Copy the frozen model file into the image. This bundles a specific
# model snapshot directly into the container -- no MLflow server needs
# to be reachable at runtime. To ship a NEW model version, run
# export_model.py locally to refresh churn_model.pkl, then rebuild
# this image. (Trade-off vs. a live registry: simpler and fully
# self-contained, but updating the model means a new image build.)
COPY churn_model.pkl .

# Document which port the app listens on (informational -- doesn't
# actually publish the port; that happens at `docker run -p`)
EXPOSE 8000

# Start the API when the container runs.
# host 0.0.0.0 is required -- 127.0.0.1 would only be reachable
# from inside the container itself, not from your machine.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]