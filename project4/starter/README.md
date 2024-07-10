# Dog Image Classification

## Overview

In this project, we will follow the steps below:

- Train and deploy a model on Sagemaker, using the most appropriate instances. Set up multi-instance training in your Sagemaker notebook.
- Adjust your Sagemaker notebooks to perform training and deployment on EC2.
- Set up a Lambda function for your deployed model. Set up auto-scaling for your deployed endpoint as well as concurrency for your Lambda function.
- Ensure that the security on your ML pipeline is set up properly.

## Steps Involved

### Step 1: Training and Deployment on SageMaker

#### Notebook Instance
Created sagemaker notebook instance I have used ml.t3.medium as this is suffiecint to run my notebook.

![Notebook Instance](./screenshots/notebook-instances.png)

#### S3 Bucket
Mention the S3 bucket used for storing data and any specific configurations.

![S3 Bucket](path/to/your/s3_bucket_image)

#### Single Instance Training
Detail the configuration for single instance training, including instance type, hyperparameters, and any special configurations.

![Single Instance Training](path/to/your/single_instance_training_image)

#### Multi-instance Training
Explain the setup for multi-instance training, including instance count and configuration details.

![Multi-instance Training](path/to/your/multi_instance_training_image)

#### Deployment
Provide details on the deployment process, including the instance type, endpoint creation, and any special configurations.

![Deployment](path/to/your/deployment_image)

### Step 2: EC2 Training

Describe the process of training the model on an EC2 instance, including the AMI used, instance type, and setup instructions.

![EC2 Training](path/to/your/ec2_training_image)

### Step 3: Lambda Function Setup

Explain the steps to set up a Lambda function for your deployed model. Include details on the function configuration, IAM role, and necessary permissions.

![Lambda Function Setup](path/to/your/lambda_function_image)

### Step 4: Lambda Security Setup and Testing

#### Security Policies
Describe the security policies attached to the Lambda function, including IAM roles, permissions, and any security groups used.

![Security Policies](path/to/your/security_policies_image)

#### Testing Lambda Function
Detail the process for testing the Lambda function, including sample requests and expected responses.

![Testing Lambda](path/to/your/testing_lambda_image)

### Step 5: Lambda Concurrency Setup and Endpoint Auto-scaling

#### Concurrency
Explain the setup for Lambda function concurrency, including reserved concurrency settings.

![Concurrency](path/to/your/concurrency_image)

#### Auto-scaling
Provide details on setting up auto-scaling for the SageMaker endpoint, including scaling policies and target metrics.

![Auto-scaling](path/to/your/auto_scaling_image)

## Conclusion

Summarize the outcomes of the project, key takeaways, and any future work or improvements that could be made.

## References

Include any references or resources used in the project.

