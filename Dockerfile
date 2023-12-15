# Use the official Lambda Python runtime
FROM public.ecr.aws/lambda/python:3.8

# Install required dependencies
RUN pip install keras-image-helper \
    && pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

# Copy necessary files
COPY kitchenware_model.tflite .
COPY lambda_function.py .

# Set environment variable for the model name
ENV MODEL_NAME=kitchenware_model.tflite

# Specify the command to run
CMD ["lambda_function.lambda_handler"]
