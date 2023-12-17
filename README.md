# Kitchenware Image Classification System with Keras: Deploying and Testing with AWS Lambda and TensorFlow Lite

![Kitchenware Image](images/kitchenware_image.png)

## Description of the Problem

In this project, we aim to develop an image classification system that can identify various kitchenware items such as cups, forks, glasses, knives, plates, and spoons. The goal is to create a model that can assist in automating the categorization of kitchenware items from images.

## How a Model Could Be Used

The trained model can be used in applications where automatic classification of kitchenware items is needed. This could include inventory management systems, smart kitchen appliances, or any scenario where the identification of kitchenware items from images is required.

### Competition Overview

The competition, which had already concluded, provided a structured framework for evaluating the performance of image classification models. It featured a diverse dataset collected using Toloka, with additional contributions shared by participants. The evaluation metric used was classification accuracy.

### Personal Goals

My objectives for the project included honing my machine learning skills, gaining practical experience in image classification, and exploring the challenges associated with making machine learning models production-ready.

### Methodology

I followed a systematic approach, including data preprocessing, model development using Keras, and fine-tuning to achieve optimal performance. I also explored additional kitchenware images contributed by participants, augmenting the main competition dataset to enhance model training.

### Results and Achievements

While the competition has concluded, my focus extended beyond achieving a high classification accuracy. I aimed to create a production-ready solution, considering factors such as model interpretability, scalability, and deployment ease.

### Learnings and Challenges

Throughout the project, I encountered and addressed various challenges, refining my understanding of image classification, model optimization, and the transition from a prototype to a production-ready solution. The experience provided valuable insights into real-world machine learning applications.

# Project Structure

- **notebook.ipynb**: A Jupyter Notebook containing detailed steps of the project, including data preparation, exploratory data analysis (EDA), feature importance analysis, model selection, and parameter tuning.

- **train.py**: A script responsible for training the final model, saving it to a file

- **predict.py**: A script to load the trained model, serve it via a web service (using AWS Lambda - serverless), and make predictions.

- **dependencies.txt**: A file listing the dependencies for the project, including library versions.

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
   **lambda_invoke_url** = 'http://localhost:9000/2015-03-31/functions/function/invocations'
   **Run the Web Service:**
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

Follow me on Twitter 🐦, connect with me on LinkedIn 🔗, and check out my GitHub 🐙. You won't be disappointed!

👉 Twitter: https://twitter.com/NdiranguMuturi1
👉 LinkedIn: https://www.linkedin.com/in/isaac-muturi-3b6b2b237
👉 GitHub: https://github.com/Isaac-Ndirangu-Muturi-749

So, what are you waiting for? Join me on my tech journey and learn something new today! 🚀🌟