import logging
import tensorflow as tf
from datetime import datetime
from model_utils import f1
from inference import run_pipeline

def predict_new_images():
    model = tf.keras.models.load_model("/app/models/deployed_model.h5", custom_objects={"f1": f1})
    input_dir = "/data/new_images"
    output_csv_path = f"/data/predictions/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    run_pipeline(input_dir, model, output_csv_path)
    logging.info("✅ New predictions generated.")
