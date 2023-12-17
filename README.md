# Kitchenware Image Classification System with Keras: Deploying and Testing with AWS Lambda and TensorFlow Lite

![Kitchenware Image](images/kitchenware_image.png)

## Description of the Problem

This project focuses on developing an image classification system capable of identifying various kitchenware items, including cups, forks, glasses, knives, plates, and spoons. The primary objective is to create a model that can automate the categorization of kitchenware items based on images.

## How a Model Could Be Used

The trained model finds application in scenarios where automatic classification of kitchenware items is essential. This includes inventory management systems, smart kitchen appliances, or any situation requiring the identification of kitchenware items from images.

### Competition Overview

The competition, which has concluded, provided a structured framework for evaluating image classification models. The dataset, collected using Toloka with additional participant contributions, employed classification accuracy as the evaluation metric.

### Personal Goals

My project objectives included refining machine learning skills, gaining practical experience in image classification, and addressing challenges associated with making machine learning models production-ready.

### Methodology

A systematic approach was adopted, covering data preprocessing, model development using Keras, and fine-tuning for optimal performance. Additional kitchenware images contributed by participants were explored, augmenting the main competition dataset for improved model training.

### Results and Achievements

While achieving high classification accuracy was a goal, the focus extended to creating a production-ready solution, considering factors like model interpretability, scalability, and deployment ease.

### Learnings and Challenges

Throughout the project, various challenges were encountered and addressed, enhancing understanding of image classification, model optimization, and transitioning from a prototype to a production-ready solution. The experience provided valuable insights into real-world machine learning applications.

# Project Structure

- **notebook.ipynb**: A Jupyter Notebook detailing project steps, including data preparation, exploratory data analysis (EDA), feature importance analysis, model selection, and parameter tuning.

- **train.py**: A script for training the final model and saving it to a file.

- **predict.py**: A script for loading the trained model, serving it via a web service (using AWS Lambda - serverless), and making predictions.

- **dependencies.txt**: A file listing project dependencies, including library versions.

- **Dockerfile**: Enables running the service in a Docker container, providing instructions for setting up the environment.

# Running the Project

## **Install Dependencies:**
```bash
pip install -r dependencies.txt
```

## **Data Download or Preparation:**

- Access the dataset at [Kaggle - Kitchenware Classification](https://www.kaggle.com/c/kitchenware-classification/data)
  - **Dataset Description:**

    This dataset comprises images representing various kitchenware items.

  **Files:**

  - `train.csv`: Training set containing Image IDs and corresponding classes.
  - `test.csv`: Test set containing Image IDs.
  - `sample_submission.csv`: Sample submission file demonstrating the correct format.
  - `images/`: Directory containing images in JPEG format.

## **Training the Model:**
Leverage services such as Saturn Cloud or Kaggle with GPU acceleration for optimal performance during the training process.
```bash
python train.py
```

# Deployment: Invoke Lambda Function for TensorFlow Lite Model Inference

## Deploy by running the Docker image locally on port 9000
   - Using Docker:
     ```bash
     docker build -t kitchenware-classification-model .
     docker run -it --rm -p 9000:8080 kitchenware-classification-model:latest
     ```
   - lambda_invoke_url = 'http://localhost:9000/2015-03-31/functions/function/invocations'  
   - Run the Web Service:  
   ```bash
   python predict.py
   ```

## Deploy on AWS Lambda and exposed through API Gateway.

The provided Python script (`predict.py`) demonstrates how to invoke a Lambda function deployed with a TensorFlow Lite model for image classification.

- **Lambda Invocation:**
  - The `invoke_lambda_function` function is used to invoke the Lambda function.
  - The Lambda function is deployed on AWS Lambda and exposed through API Gateway.
  - lambda_invoke_url = `'https://9vmhopr686.execute-api.eu-west-3.amazonaws.com/testAPI/predict'`

- **Image URLs:**
  - Two test image URLs are provided for inference:
    1. A cup image: [Link to Image 1](https://th.bing.com/th/id/OIP.LdDqEwKlL3wJ1zFPjzvFqwHaIW?rs=1&pid=ImgDetMain)
    2. A glass image: [Link to Image 2](https://th.bing.com/th/id/R.72f44a5975ee79e83703cfb10809f0be?rik=6ht3ChPx7G84yA&pid=ImgRaw&r=0)

   **Run the Web Service:**
   ```bash
   python predict.py
   ```

Follow me on Twitter

 🐦, connect with me on LinkedIn 🔗, and check out my GitHub 🐙. You won't be disappointed!

👉 Twitter: https://twitter.com/NdiranguMuturi1  
👉 LinkedIn: https://www.linkedin.com/in/isaac-muturi-3b6b2b237  
👉 GitHub: https://github.com/Isaac-Ndirangu-Muturi-749  

So, what are you waiting for? Join me on my tech journey and learn something new today! 🚀🌟