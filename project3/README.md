# Image Classification using AWS SageMaker

Capstone Project: Fine-Tuning a Pretrained Model with AWS SageMaker

## Project Overview

This project demonstrates how to use Amazon SageMaker to build, train, and deploy an image classification model. The model is based on a pre-trained ResNet50 network from the torchvision library, fine-tuned for a specific classification task.

## Project Objectives and Outcomes

- **Model Fine-Tuning**: Choose a pre-trained model, such as ResNet50, and apply transfer learning techniques to adapt it to your chosen dataset.
- **SageMaker Features**: Implement SageMaker's profiling and debugging tools to monitor model training and performance. Conduct hyperparameter tuning to optimize your model's performance.
- **Model Deployment**: Deploy your fine-tuned model to a SageMaker endpoint. Ensure that you can query the deployed model with a sample image to retrieve a prediction.


## Setup Instructions

### Prerequisites

- AWS Account
- AWS CLI configured with appropriate permissions
- Python 3.x environment

### Installation

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Upload Data to S3**:
    Ensure your dataset is available in an S3 bucket. If not, upload it using the AWS CLI or S3 console.

2. **Run the Notebook**:
    Open and run the `train_and_deploy.ipynb` notebook. This notebook will guide you through the entire process of setting up your data, training the model, tuning hyperparameters, and deploying the model.

3. **Train the Model**:
    Ensure the `train_model.py` script is properly configured and contains all the necessary code to train your model.

4. **Deploy the Model**:
    Follow the steps in the notebook to deploy the model to a SageMaker endpoint.

### Running the Code

1. **Data Preparation**:
    ```python
    # Command to download and unzip data
    !wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
    !unzip dogImages.zip -d /tmp/
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/dogImages/train', '<your-bucket-name>', 'train')
    s3.upload_file('/tmp/dogImages/test', '<your-bucket-name>', 'test')
    ```

2. **Training and Tuning**:
    ```python
    # Setup and start hyperparameter tuning job
    tuner.fit({'train': f's3://<your-bucket-name>/train', 'test': f's3://<your-bucket-name>/test'})
    ```

3. **Model Deployment**:
    ```python
    predictor = best_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
    ```

4. **Query the Endpoint**:
    ```python
    image_path = '/path/to/sample/image.jpg'
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    response = predictor.predict(image_array)
    print(response)
    ```

5. **Cleanup**:
    ```python
    predictor.delete_endpoint()
    ```

## Results

- **Model Performance**: Provide a summary of the model's performance, including accuracy, loss, and any other relevant metrics.
- **Hyperparameter Tuning Results**: Describe the best hyperparameters found during tuning.
- **Deployment**: Confirm the successful deployment of the model and provide examples of predictions made by the model.

## Conclusion

Summarize the key takeaways from the project, including any challenges faced and how they were overcome. Reflect on the skills and knowledge gained during the project.

## References

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
