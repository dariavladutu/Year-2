# Block D - Engineer - MLOps
### Year 2 | Applied Data Science & AI | Breda University of Applied Sciences

---

## Block Overview

In Blocks B and C, students built machine learning models for computer vision and natural language processing, focusing on the modelling component of the CRISP-ML cycle. These models served as proof of concepts that demonstrated the ability to solve specific problems. However, most of these models were not production-ready.

Block D teaches students how to **productionise and deploy** machine learning models to a cloud platform. Students learn how to monitor their models and ensure continuous retraining on new data. The block covers creating robust codebases, establishing data ingestion and model training pipelines, devising deployment approaches, implementing monitoring strategies, and demonstrating deployed models in action.

**Duration**: 10 weeks (Weeks 1-10)  
**Project Due**: Friday 27th June 2025 at 16:59  
**Methodology**: Agile Scrum (team-based, groups of 5 students)

---

## Learning Objectives

By the end of this block, students will be able to:

- Understand the concept of MLOps and the role of DevOps in the ML lifecycle
- Create a robust production-ready codebase and establish data ingestion and model training pipelines
- Productionise and deploy model(s) in data-driven applications locally, to on-premise servers, and to a cloud platform using different deployment strategies
- Monitor model(s) and ensure that they are continuously retrained on new data
- Create professional documentation for projects

---

## Project Options

Students select one of two project options and develop a production-grade model that can be deployed on a cloud platform:

### Option 1: Computer Vision and Robotics

Computer vision plays a pivotal role in plant science and many other fields. The HADES system at The Netherlands Plant Eco-phenotyping Centre (NPEC) generates large amounts of image data for plant science research. Computer vision models can automate the analysis of this data, and when combined with robotics, can automate the entire plant phenotyping research process.

In Block B, students built proof-of-concept computer vision models for object detection, segmentation, localisation, and measurement, combined with a robotic simulation for automated inoculation. In Block D, the task is to productionise and deploy these models to a cloud platform as an application or service.

### Option 2: Natural Language Processing

Natural language is a language that has evolved naturally as a means of communication among people. Computer analysis of natural language is called Natural Language Processing (NLP). NLP is used for email classification, smart assistants, search engines, language translation, and many more applications.

As a producer of several TV shows, Banijay Benelux and their partners at Content Intelligence Agency want better understanding of what components of TV shows gain more attention than others. They want to gather data on episodes to understand which parts perform well with viewers and which are less interesting.

In Block C, students built proof-of-concept NLP models for classifying emotions in a TV series. In Block D, the task is to productionise and deploy these models to a cloud platform as an application or service.

---

## General Project Requirements

The project will be completed in groups of 5 students. Students must form their own groups; those unable to form a group will be assigned to one.

Students must select one of the given options and develop a production-grade model that can be deployed on a cloud platform. Tasks include:

- Creating a robust codebase
- Establishing data ingestion and model training pipelines
- Devising a model deployment approach
- Implementing a monitoring strategy
- Demonstrating the deployed model(s) in action

**General Requirements:**

- Develop a scalable API that accepts input data and returns predictions along with confidence scores
- Ensure that the model supports automatic retraining on a weekly basis or when new data is available
- Continuously monitor prediction accuracy to prevent model drift
- Maintain clear records of the deployed model and its training process for auditing purposes (data, code, and model versioning)
- Minimize manual intervention in deployment, with manual approval required for production (e.g., via a CI/CD pipeline, automated testing, and deployment)
- Store and manage code in a code repository, such as GitHub

---

## Project Deliverables

**Due: Friday 27th June 2025 at 16:59**

- A GitHub repository containing all code and documentation for the project
- A README.md file containing project description, installation instructions, and usage guide
- Installable Python package(s) containing the code (include wheel file and source distribution)
- A Docker container containing training and inference code and an API for the model(s), runnable locally
- Microsoft Azure ML workspace containing the model(s)
- A demo of the deployed model/s - to be shown on demo day (last Friday of the block)
- A product backlog and sprint backlog, including user stories, tasks, and estimates
- Architecture diagrams for the project, including data pipeline, model training pipeline, and deployment pipeline
- A project plan and road map including milestones and deadlines
- Learning log, work log, and completed peer reviews (all must be evidenced in learning log)

---

## Team-Based Working - Agile Scrum

For Block D, students are expected to follow an agile project management methodology called **Scrum**.

The Scrum methodology is characterized by short phases called "sprints" wherein project work occurs. During sprint planning, the project team identifies a small part of the scope—a set of tasks to be completed during the upcoming sprint, which is usually a two-week period.

At the end of the sprint, this work should be ready to be delivered to the client. Finally, the sprint ends with a sprint review and retrospective (or rather, lessons learned). This cycle is repeated throughout the project lifecycle until the entirety of the scope has been delivered or Block D is at an end.

### Azure DevOps Tools

In this block, students will use Azure DevOps to manage projects and facilitate agile project management methodology. The following Azure DevOps features will be used:

- **Boards** - To manage project backlog and tasks, plan sprints, and track progress
- **Task Estimating** - To estimate the effort required for each task and plan sprints more effectively
- **Retrospectives** - To reflect on progress and identify areas for improvement

---

## Medal Challenges

The medal courses for this block have been specifically selected to help boost portfolios and extend knowledge and expertise. Students may finish these after the initial deadline.

GitHub Medals will be awarded with the following criteria:

- **🏅 Blog Post**: Write a blog post describing and comparing the MLOps tools offered by AWS, Azure, and GCP. Include a comparison of pricing models and the pros and cons of each.

- **🏅 Streamlit/Gradio App**: Create a Streamlit or Gradio app that allows users to upload data, make predictions using a deployed cloud model, and visualize results using an explainable AI tool.

- **🏅 Best Deployed Application**: The best deployed application judged by staff and students on Demo Day will receive a gold medal.

---

## Block Outline

### Sprint 1 - Introduction to MLOps, Project Briefing and Project Planning and Scoping
**Weeks: 1-2**

Introduction to the concept of MLOps and the tools and technologies used to productionise and deploy machine learning models. Students will be briefed on the project and begin planning and scoping their projects.

**Goals for this Sprint:**
- Understand the concept of MLOps and the tools, technologies, and frameworks used to productionise and deploy machine learning models
- Understand the project requirements and deliverables
- Create a project plan and road map for the project (think about all features required and how to implement them, then prioritise)
- Create a plan for the Python package
- Understand and apply Git and GitHub best practices for collaboration (branching, merging, pull requests, etc.)
- Set up project repository on GitHub with correct folder structure, appropriate README.md files, and branching strategy with protected branches
- Set up package management and virtual environments for the project

---

### Sprint 2 - MVP Inference Application
**Weeks: 3-4**

Create production-ready code for a minimum viable product application that can perform inference. Learn how to write clean, modular, well-documented code that conforms to industry best practices. Learn how to use logging and testing to ensure code robustness and reliability. Build an API for the model, containerise the application, and deploy it to a server.

**Goals for this Sprint:**
- Create a complete, well-documented, and tested MVP Python package that can be installed and used by others
- Use logging to track code progress and help with debugging
- Use unit testing to ensure code robustness and reliability
- Create a CLI to enable users to interact with the package (inference only) from the command line without writing code
- Use docstrings in combination with Sphinx to create professional documentation for the package
- Create an API for the model that can be used for inference
- Containerise the application using Docker
- Test the application locally using Docker
- Deploy the application to an on-premise server

---

### Sprint 3 - Data Pipelines and Model Training in the Cloud
**Weeks: 5-6**

Focus on adding features to the application and creating data pipelines and model training pipelines. Create versioned models, data assets, and environments. Learn how to automate the model training process.

**Goals for this Sprint:**
- Host and manage data in the cloud using the Azure ML Python SDK
  - Upload data to a datastore in Azure ML
  - Create versioned data assets (e.g., train, test, validation sets)
- Manage environments and dependencies in the cloud using the Azure ML Python SDK
  - Create and register environments in Azure ML that contain dependencies for training, evaluation, and inference code
  - Test the environments with a simple training job with a small subset of data
- Create a data pipeline and model training pipeline in the cloud. The pipeline should (at the very least):
  - Ingest data from the data assets created
  - Preprocess the data
  - Train a model
  - Evaluate the model
  - Register the model if it meets evaluation criteria
- Create a basic frontend for the application allowing users to interact with the model. Containerise the frontend and deploy it to a server. Use Streamlit, Gradio, or another frontend framework of choice.
- Integration of data pipeline and model training pipeline with the application

**Optional:**
- Create a pipeline for automated hyperparameter tuning
- Schedule training pipeline to run on a regular basis or based on a trigger

---

### Sprint 4 - Automated Deployment, Monitoring, and Continuous Retraining
**Weeks: 7-8**

Focus on model deployment in the cloud, monitoring, and adding advanced features. Learn how to deploy models to a cloud platform and use what has been learned to deploy and monitor models in the cloud.

**Goals for this Sprint:**
- Deploy model to the cloud using multiple deployment methods/strategies
- Setup monitoring for the model
- Automate the deployment of the model using a CI/CD pipeline
- Setup continuous retraining for the model, including the following:
  - Allow users to give feedback on predictions made by the model
  - Use the feedback to retrain the model
  - Create a retraining pipeline that can be triggered by new data or user feedback
- Create a monitoring dashboard for the model/application

---

### Sprint 5 - Testing and Evaluation
**Weeks: 9-10**

Focus on testing and evaluation. Test the deployment and evaluate its performance. Create a demo of the model and wrap up the project.

**Goals for this Sprint:**
- Complete any outstanding tasks from previous sprints. Prioritise tasks critical to project success.
- Test deployment and evaluate its performance
  - Test with synthetic data and requests
  - Run real-world user tests
- Create a demo of the deployed model/application. Including the following:
  - A deployed application or service that uses the model that users can interact with (e.g., a web app, a REST API, etc. - at absolute minimum the fast API docs should be available, but a frontend is preferred) - **SCREEN 1**
  - Documentation for the model and how to use it - **SCREEN 2**
  - A poster describing the application and deployment - **SCREEN 3**
  - Any additional evidence that can be shown (coverage reports, GitHub actions, Azure ML Workspace, etc.) - **SCREEN 4**

**⚠️ Remember:** The demo will form part of the grading process. It is an opportunity to put your best foot forward and showcase work in the best light possible. It is easier for staff to assess students who are present during the process and have the opportunity to explain their work. **⚠️**

---

## Staff Members

| Name | Availability | Week | Email | Expertise |
|------|--------------|------|-------|-----------|
| Dean van Aswegen, MSc. | Mon - Fri | Weeks 1-8 | aswegen.d@buas.nl | MLOps, Robotics, Reinforcement Learning |
| Jason Harty, BSc. | Mon - Fri | Weeks 1-8 | harty.d@buas.nl | Data Engineering, Robotics |
| Alican Noyan, PhD. | Mon-Thu | Weeks 1-8 | noyan.a@buas.nl | Computer Vision, NLP |
| Myrthe Buckens, MA. | Mon, Tue, Wed, Thu | Weeks 1-8 | buckens.m@buas.nl | Natural Language Processing |
| Shival Indermun, PhD. | Mon - Fri | Weeks 1-8 | indermun.s@buas.nl | Computer Vision, Robotics |
| Karna Rewatkar | Mon - Fri | Weeks 1-8 | rewatkar.k@buas.nl | Computer Vision |
| Tsegaye Tashu, PhD. | Mon, Tue, Wed, Thu | Weeks 1-8 | tashu.t@buas.nl | Natural Language Processing |
| Frank Peters, PhD. | Mon-Thu | Omnipresent | peters.f@buas.nl | Omnipotent |

---

## Suggested Learning Resources

### Sprint 1
- [Intro to MLOps (Machine Learning Operations)](https://learn.microsoft.com/en-us/training/paths/introduction-machine-learn-operations/)
- [Git and GitHub Best Practices for collaboration](https://learn.microsoft.com/en-us/training/modules/github-introduction/)
- [Developing Python Packages](https://learn.microsoft.com/en-us/training/modules/python-packages/)
- [Production Ready Code Cheat Sheet](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- [MLOps Maturity Levels - Article from MS Azure ML Accelerator](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)
- [MLOps Maturity Model with Architecture Diagrams](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-technical-paper)
- [Virtual environments & Package management with Python](https://realpython.com/python-virtual-environments-a-primer/)
- [Production Ready Code with Python](https://realpython.com/python-application-layouts/)
- [Working with Azure DevOps](https://learn.microsoft.com/en-us/training/modules/get-started-with-devops/)
- [Project and Package Planning](https://learn.microsoft.com/en-us/training/modules/python-create-package/)

### Sprint 2
- [MLOps Maturity Levels - Article from MS Azure ML Accelerator](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)
- [Client-side Continuous Integration Tools with Python](https://realpython.com/python-continuous-integration/)
- [Building Packages for Users - CLIs and APIs](https://realpython.com/python-cli-applications/)
- [Building and Distributing Python Packages](https://realpython.com/python-wheels/)
- [Containerisation with Docker](https://docs.docker.com/get-started/)
- [Application Programming Interfaces (APIs)](https://realpython.com/api-integration-python/)
- [Intro to CI/CD - Automated Testing](https://learn.microsoft.com/en-us/training/modules/explain-devops-continous-delivery-quality/)
- [On-Premise Deployment](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/apps/devops-dotnet-baseline)
- [Testing and Logging with Python](https://realpython.com/python-testing/)
- [Documentation for Python](https://realpython.com/documenting-python-code/)

### Sprint 3
- [Intro to the Cloud and MLOps Tools](https://learn.microsoft.com/en-us/training/paths/introduction-machine-learn-operations/)
- [Microsoft Azure Machine Learning I](https://learn.microsoft.com/en-us/training/paths/build-ai-solutions-with-azure-ml-service/)
- [Data Modeling and Storage](https://learn.microsoft.com/en-us/training/modules/design-data-storage-solution-for-relational-data/)
- [Data Pipeline Design](https://learn.microsoft.com/en-us/training/modules/design-data-integration/)
- [Microsoft Azure Machine Learning II](https://learn.microsoft.com/en-us/training/paths/train-models-azure-machine-learning/)
- [MLFlow](https://mlflow.org/docs/latest/index.html)

### Sprint 4
- [ML Application Deployment Options and Strategies](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-technical-paper)
- [Azure Machine Learning III - Deployment](https://learn.microsoft.com/en-us/training/paths/deploy-manage-models-azure-machine-learning/)
- [Advanced GitHub and CI/CT/CD](https://learn.microsoft.com/en-us/training/modules/github-actions-ci/)
- [Azure Machine Learning IV - Monitoring and Testing](https://learn.microsoft.com/en-us/training/paths/monitor-azure-machine-learning/)
- [Continuous Retraining - Data Flywheels](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-technical-paper)

---

## Important Notes

**⚠️ Note**: The timeline above is indicative and may change. The dates are not fixed and may be adjusted based on the progress of individual groups at the discretion of the product owner. Currently, the plan looks very waterfall, but this will evolve slightly differently for each group as progress is made through the block.

**⚠️ Before sprint planning meetings**: Please work on any outstanding issues from the previous sprint.

---

## Additional Information

For more details on the project options, please see:
- [Creative Brief Computer Vision](link)
- [Creative Brief Natural Language Processing](link)

**Contact**: Frank Peters, PhD. (peters.f@buas.nl)

---

*Applied Data Science and Artificial Intelligence @ Breda University of Applied Sciences*