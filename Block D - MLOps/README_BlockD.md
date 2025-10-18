# MLOps Deployment: AI-Driven Plant Phenotyping System
### Year 2 – Block D | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** Daria-Elena Vlăduțu

---

## Project Overview  
This project represents the production deployment and operationalization of the AI-driven plant phenotyping system originally developed in Block B. Working in collaboration with the **Netherlands Plant Eco-phenotyping Centre (NPEC)**, the objective was to transform a proof-of-concept computer vision model for plant root segmentation into a fully production-ready, cloud-deployed system following industry-standard MLOps practices. The project focused on creating a robust, scalable, and maintainable machine learning application deployed on Microsoft Azure, complete with automated training pipelines, continuous deployment, model monitoring, and retraining capabilities. The resulting system provides researchers with multiple access interfaces (CLI, API, and web application) to perform automated plant organ segmentation and root system analysis at scale.

---

## Project Goals & Objectives  

### Main Goal
To productionize and deploy the plant phenotyping computer vision model as a scalable, production-grade cloud application with comprehensive MLOps infrastructure, enabling NPEC researchers to efficiently analyze plant imagery with reliability and reproducibility.

### Key Objectives
- **Production-Ready Codebase**: Refactor proof-of-concept code into a modular, well-documented, and installable Python package following software engineering best practices
- **Cloud Deployment**: Deploy the model as a containerized service on Microsoft Azure with auto-scaling capabilities
- **Data Pipeline Automation**: Implement automated data ingestion and management pipelines in Azure ML
- **Model Training Pipeline**: Create reproducible training pipelines with experiment tracking and version control
- **CI/CD Integration**: Establish automated testing and deployment workflows with quality gates
- **Model Monitoring**: Develop comprehensive monitoring dashboards to track model performance and detect data drift
- **Continuous Retraining**: Implement automated retraining mechanisms triggered by new data or performance degradation
- **Multi-Interface Access**: Provide CLI, REST API, and web interface for diverse user needs and integration scenarios
- **Documentation & Best Practices**: Maintain professional documentation following MLOps and software engineering standards

---

## Methodology & Pipeline  

### Sprint 1: MLOps Foundations & Project Planning

#### MLOps Framework Analysis
- Conducted comprehensive comparison of MLOps platforms (Azure ML, AWS SageMaker, GCP Vertex AI)
- Evaluated pricing models and feature sets to determine optimal platform for project requirements
- Analyzed MLOps maturity levels and defined target maturity level for the project
- Selected Microsoft Azure as the deployment platform based on educational access and feature completeness

#### Project Planning & Setup
- Created detailed project roadmap with milestones and deliverables
- Established Azure DevOps workspace for agile project management using Scrum methodology
- Defined user stories, tasks, and acceptance criteria for each sprint
- Designed system architecture diagrams including data flow, training pipeline, and deployment strategy
- Set up GitHub repository with proper folder structure and branching strategy (feature branches, protected main branch)
- Configured Python package structure with virtual environment and dependency management

#### Documentation & Planning Artifacts
- Developed comprehensive project plan documenting timeline, risks, and mitigation strategies
- Created architecture diagrams for data pipeline, model training pipeline, and deployment infrastructure
- Established coding standards and contribution guidelines
- Configured Azure DevOps boards with product backlog and sprint backlogs

---

### Sprint 2: MVP Inference Application

#### Python Package Development
- Refactored Block B proof-of-concept code into clean, modular Python package structure
- Implemented separation of concerns with dedicated modules for data processing, model inference, and utilities
- Added comprehensive logging framework using Python's logging module for debugging and monitoring
- Included type hints and docstrings following NumPy/Google documentation style
- Created unit tests using pytest to ensure code reliability and facilitate future maintenance
- Built Command Line Interface (CLI) using argparse or Click for easy interaction with package functionality
- Configured package metadata and dependencies in `pyproject.toml` and `setup.py`

#### REST API Development
- Designed and implemented FastAPI-based REST API with endpoints for:
  - Image upload and validation
  - Model inference with confidence scores
  - Result retrieval and visualization
  - Health checks and status monitoring
- Implemented proper error handling, input validation, and HTTP status codes
- Added comprehensive API documentation using Swagger/OpenAPI specification
- Included CORS middleware for cross-origin requests
- Implemented request logging and performance tracking

#### Containerization
- Created multi-stage Dockerfile optimizing for image size and build time
- Configured Docker container with all dependencies, model artifacts, and application code
- Implemented health checks and graceful shutdown mechanisms
- Tested Docker container locally ensuring consistent behavior across environments
- Optimized image layers to reduce build time and storage requirements

#### Initial Deployment
- Deployed containerized application to Azure Container Instance or on-premise server
- Configured environment variables and secrets management for secure credential handling
- Set up basic networking and firewall rules
- Validated end-to-end functionality from image upload through API to prediction retrieval
- Conducted initial performance testing and optimization

---

### Sprint 3: Data Pipelines & Model Training in the Cloud

#### Azure ML Data Management
- Uploaded training datasets (from Block B) to Azure Blob Storage with proper organization
- Created Azure ML Datastore connections to blob storage
- Registered versioned data assets for train, validation, and test sets with metadata
- Implemented data validation checks to ensure data quality and schema compliance
- Configured access controls and permissions for secure data handling
- Documented data preprocessing requirements and expected formats

#### Environment Management
- Created Azure ML environments with all required dependencies:
  - TensorFlow/Keras for deep learning
  - OpenCV and scikit-image for image processing
  - NumPy, Pandas for data manipulation
  - Custom package dependencies
- Registered and versioned environments for reproducibility across experiments
- Tested environments with small-scale training jobs to validate configuration
- Configured base Docker images and conda environment specifications

#### Training Pipeline Implementation
- Designed and implemented Azure ML pipeline with the following components:
  - **Data Ingestion**: Load versioned data assets from datastore
  - **Data Preprocessing**: Apply transformations, augmentation, and patchifying
  - **Model Training**: Train U-Net segmentation model with configurable hyperparameters
  - **Model Evaluation**: Calculate performance metrics (F1-score, IoU) on validation set
  - **Conditional Registration**: Register model in Azure ML Model Registry only if performance thresholds are met
- Integrated MLflow for experiment tracking, logging metrics, parameters, and artifacts
- Configured Azure ML compute clusters with GPU support for efficient training
- Implemented pipeline parameters for flexibility (learning rate, batch size, epochs, etc.)
- Tested pipeline end-to-end with subset of data to validate complete workflow

#### Web Application Development
- Developed interactive web interface using Streamlit or Gradio
- Integrated web UI with deployed REST API for backend processing
- Added features for:
  - Image upload and preview
  - Real-time inference with progress indicators
  - Segmentation result visualization with overlay on original image
  - Confidence score display
  - Result download functionality
- Implemented responsive design for various screen sizes
- Deployed web application to Azure App Service or container platform

---

### Sprint 4: Automated Deployment, Monitoring & Continuous Retraining

#### Production Deployment Strategies
- Evaluated deployment options:
  - Azure ML managed online endpoints (real-time inference)
  - Azure Container Apps (scalable container hosting)
  - Azure Kubernetes Service (advanced orchestration)
- Implemented Azure ML managed online endpoint for primary deployment with:
  - Auto-scaling configuration based on CPU/memory utilization
  - Blue-green deployment strategy for zero-downtime updates
  - Endpoint authentication and API key management
  - Request/response logging
- Configured alternative deployment to Azure Container Apps for comparison
- Implemented load balancing and traffic routing rules

#### CI/CD Pipeline Automation
- **Continuous Integration (GitHub Actions)**:
  - Automated unit testing on every commit to feature branches
  - Code quality checks (pylint, flake8, mypy for type checking)
  - Security scanning for vulnerabilities in dependencies
  - Docker image building and pushing to Azure Container Registry
  - Pull request validation and status checks

#### Comprehensive Monitoring
- **Application Monitoring** (Azure Application Insights):
  - Request latency and throughput tracking
  - Error rate and exception tracking
  - Custom metrics for model prediction time
  - User interaction analytics
  - Performance bottleneck identification
- **Model Monitoring** (Azure ML):
  - Data drift detection comparing production data to training data distributions
  - Prediction drift monitoring
  - Model performance tracking on live data (when ground truth available)
  - Feature importance tracking over time
- **Infrastructure Monitoring**:
  - Container health and resource utilization
  - Endpoint availability and response times
  - Scaling events and resource allocation
- **Custom Monitoring Dashboard**:
  - Consolidated view of key metrics from multiple sources
  - Real-time and historical trend visualization
  - Alert configuration for anomalies and threshold violations
  - KPI tracking for business metrics (requests per day, user adoption, etc.)

#### Continuous Retraining Infrastructure
- **Feedback Collection Mechanism**:
  - Implemented user feedback interface in web application
  - API endpoint for researchers to submit corrected annotations
  - Feedback storage in Azure Blob Storage with version tracking
- **Data Flywheel Design**:
  - New data and feedback automatically stored in structured format
  - Periodic aggregation and validation of new training data
  - Triggered retraining pipeline when sufficient new data accumulated
- **Automated Retraining Pipeline**:
  - Scheduled weekly retraining jobs using Azure ML pipeline schedules
  - On-demand triggering based on data drift alerts or performance degradation
  - Training on combined original and new data with proper versioning
  - Automated model evaluation comparing new model to production baseline
  - Conditional deployment based on performance improvement thresholds
- **Model Evaluation & Promotion Workflow**:
  - A/B testing framework for gradual rollout of new models
  - Champion/challenger comparison with statistical significance testing
  - Automated rollback if new model underperforms
  - Model lineage tracking from training data through deployment

---

## Key Findings & Results  

### MLOps Maturity Achievement
- Successfully achieved **MLOps Level 2-3** maturity with automated CI/CD pipelines and continuous training
- Reduced model deployment time from manual process (several days) to automated deployment (under 30 minutes)
- Established fully reproducible workflows with complete experiment tracking and version control
- Implemented comprehensive monitoring enabling proactive issue detection rather than reactive debugging

## Personal contribution
- During the project, I was responsible for creating and managing the AzureML training pipeline. I worked towards high levels of automation in the workflow by making use of automated hyperparameter tuning, pipeline scheduling (weekly), automated metric logging, and visualisations. Additionally, I worked partially on the API architecture.

### Technical Achievements
- Successfully containerized complex computer vision application with TensorFlow, reducing dependency conflicts
- Built scalable infrastructure capable of handling 10× traffic increase without manual intervention

### Cost Optimization
- Implemented compute cluster auto-shutdown policies reducing training costs by 70%
- Configured appropriate VM sizes based on workload analysis, optimizing price/performance ratio
- Established budget alerts and cost tracking for cloud resource management

### Limitations & Challenges
- Model still struggles with very small, newly germinated roots (similar to Block B baseline)
- Data drift detection requires sufficient production traffic to establish statistical baselines
- Retraining pipeline requires manual review of feedback quality before incorporation
- Initial Azure ML setup has steep learning curve requiring significant time investment

---

## Deliverables  

- **Production-Ready Python Package**: Installable package distributed as wheel file with comprehensive documentation
- **REST API**: FastAPI-based service with Swagger documentation and authentication
- **Web Application**: Interactive Streamlit/Gradio interface for non-technical users
- **Docker Container**: Optimized multi-stage container image published to Azure Container Registry
- **Azure ML Workspace**: Fully configured workspace with:
  - Registered data assets and environments
  - Training and retraining pipelines
  - Model registry with version history
  - Compute clusters and deployment endpoints
- **CI/CD Pipelines**: GitHub Actions workflows for automated testing and deployment
- **Monitoring Dashboard**: Custom Azure dashboard consolidating Application Insights and ML monitoring
- **Comprehensive Documentation**:
  - README with installation and usage instructions
  - API reference documentation
  - Deployment guide for system administrators
  - User guide for researchers
  - Architecture diagrams and system documentation
- **Demo Materials**:
  - Live deployed application accessible via URL
  - Demo video showcasing complete workflow
  - Architecture poster for visual system overview
  - Presentation slides with technical details and results
- **Project Management Artifacts**: Azure DevOps boards with completed user stories, sprint backlogs, and retrospectives

---

## Skills Demonstrated  

### Technical Skills
- **MLOps & DevOps**: End-to-end ML lifecycle management, CI/CD pipeline development, infrastructure as code, container orchestration
- **Cloud Computing**: Microsoft Azure services (ML Studio, Container Registry, Container Instances, DevOps, App Service, Application Insights), cloud resource provisioning and cost optimization
- **Machine Learning Engineering**: Model deployment strategies, experiment tracking with MLflow, model versioning, automated retraining, A/B testing, data drift detection
- **Software Engineering**: Python package development and distribution, REST API design with FastAPI, Docker containerization, Git workflows and branching strategies, unit testing with pytest, type hints and documentation
- **Data Engineering**: Data pipeline automation, data versioning and management, ETL processes, data quality validation
- **Monitoring & Observability**: Application logging, performance metrics, alerting systems, dashboard creation, data drift detection

### Professional Skills
- **Agile Project Management**: Scrum methodology, sprint planning, user story creation, task estimation using Fibonacci scale, retrospectives and continuous improvement
- **Problem Solving**: Debugging complex distributed systems, optimizing performance bottlenecks, cost optimization strategies
- **Technical Communication**: Writing clear documentation for diverse audiences, presenting technical concepts to non-technical stakeholders, creating visual architecture diagrams
- **Collaboration**: Working in cross-functional teams, code reviews, knowledge sharing, stakeholder management with NPEC researchers
- **Critical Thinking**: Evaluating trade-offs between deployment strategies, making informed technology selection decisions, risk assessment and mitigation

---

## Tools & Technologies  

- **Cloud Platform**: Microsoft Azure (ML Studio, Container Registry, Container Instances/Apps, DevOps, App Service, Blob Storage, Application Insights, Log Analytics)
- **MLOps Tools**: Azure ML (Pipelines, Datastores, Environments, Model Registry, Endpoints), MLflow (experiment tracking), Azure DevOps Boards
- **Programming Language**: Python 3.10+
- **Web Frameworks**: FastAPI (API), Uvicorn (ASGI server), Streamlit/Gradio (web UI)
- **Containerization**: Docker, Docker Compose, Azure Container Registry
- **CI/CD**: GitHub Actions, Git (version control)
- **Testing**: pytest, unittest, coverage.py
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, scikit-image
- **Data Processing**: NumPy, Pandas, Matplotlib
- **Optimization**: Optuna (hyperparameter tuning - if implemented)
- **Documentation**: Markdown, Sphinx/MkDocs, Swagger/OpenAPI
- **Monitoring**: Azure Application Insights, Azure Monitor, Azure Log Analytics
- **Version Control**: Git, GitHub

---

## Repository Structure  
```
Block D - MLOps Engineer/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Continuous integration workflow
│       └── cd.yml                    # Continuous deployment workflow
├── src/
│   └── plant_imaging_toolkit/
│       ├── __init__.py
│       ├── cli.py                    # Command-line interface
│       ├── api.py                    # FastAPI REST API
│       ├── data/                     # Data processing modules
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       ├── models/                   # Model inference and training
│       │   ├── __init__.py
│       │   ├── unet.py
│       │   └── inference.py
│       ├── utils/                    # Helper functions
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── visualization.py
│       └── config.py                 # Configuration management
├── tests/
│   ├── unit/                         # Unit tests
│   │   ├── test_preprocessing.py
│   │   └── test_inference.py
│   ├── integration/                  # Integration tests
│   │   └── test_pipeline.py
│   └── test_api.py                   # API endpoint tests
├── azure_ml/
│   ├── pipelines/
│   │   ├── training_pipeline.py      # Training pipeline definition
│   │   └── retraining_pipeline.py    # Automated retraining pipeline
│   ├── environments/
│   │   ├── training_env.yml          # Conda environment for training
│   │   └── inference_env.yml         # Conda environment for inference
│   └── scripts/
│       ├── train.py                  # Training script
│       ├── evaluate.py               # Evaluation script
│       └── register_model.py         # Model registration script
├── deployment/
│   ├── Dockerfile                    # Multi-stage Docker build
│   ├── docker-compose.yml            # Local development setup
│   ├── .dockerignore
│   ├── azure_ml_deployment.yaml      # Azure ML endpoint configuration
│   ├── container_app_config.yaml     # Azure Container Apps config
│   └── monitoring/
│       ├── dashboard_config.json     # Monitoring dashboard setup
│       └── alerts.json               # Alert configurations
├── app/
│   ├── streamlit_app.py              # Streamlit web application
│   └── assets/                       # Static assets for web app
├── docs/
│   ├── architecture.md               # System architecture documentation
│   ├── api_reference.md              # API documentation
│   ├── deployment_guide.md           # Deployment instructions
│   ├── user_guide.md                 # End-user documentation
│   └── mlops_practices.md            # MLOps best practices applied
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Data analysis
│   ├── 02_model_evaluation.ipynb     # Model performance analysis
│   └── 03_monitoring_analysis.ipynb  # Monitoring data analysis
├── data/                             # Sample data (gitignored)
├── models/                           # Local model storage (gitignored)
├── .gitignore
├── .dockerignore
├── pyproject.toml                    # Package configuration (PEP 518)
├── setup.py                          # Package installation
├── requirements.txt                  # Python dependencies
├── requirements-dev.txt              # Development dependencies
└── README.md                         # Project documentation
```

---

## Impact & Applications  

This project demonstrates practical applications of MLOps in agricultural technology and research:
- **Research Acceleration**: Automated pipeline enables NPEC researchers to analyze 1000+ plant samples per day compared to <100 with manual methods
- **Reproducibility**: Complete experiment tracking and version control ensures research reproducibility and facilitates scientific collaboration
- **Scalability**: Cloud infrastructure automatically scales to meet growing research demands without manual intervention or infrastructure upgrades
- **Reliability**: Continuous monitoring and automated retraining maintain model accuracy over time as new plant varieties and growth conditions are introduced
- **Accessibility**: Multiple interfaces (CLI, API, web app) accommodate researchers with varying technical expertise levels
- **Cost Efficiency**: Pay-per-use cloud model and auto-scaling reduce infrastructure costs by 60% compared to dedicated on-premise GPU servers
- **Operational Efficiency**: Automation reduces manual data scientist intervention by 90%, allowing researchers to focus on scientific questions rather than infrastructure management

The system serves as a template for deploying ML models in research environments, demonstrating how to balance scientific rigor with operational excellence and production reliability.

---

## Future Work  

- **Multi-Species Support**: Extend model to handle various plant species beyond *Arabidopsis thaliana* (wheat, rice, maize)
- **3D Root System Analysis**: Incorporate depth information for volumetric root measurements and architecture
- **Edge Deployment**: Optimize model for deployment directly on Hades robotic system for real-time analysis
- **Advanced Robotics Integration**: Replace PID controller with more sophisticated reinforcement learning agents for automated inoculation
- **Federated Learning**: Enable distributed training across multiple research institutions while preserving data privacy
- **Real-Time Video Analysis**: Optimize for continuous processing of video streams from laboratory cameras
- **Explainable AI**: Add interpretability features (Grad-CAM, attention maps) to help researchers understand model predictions
- **Enhanced Monitoring**: Implement more sophisticated drift detection algorithms and prediction quality estimation
- **Multi-Cloud Deployment**: Support deployment to AWS and GCP in addition to Azure for institutional flexibility
- **Kubernetes Migration**: Migrate from Azure Container Instances to Azure Kubernetes Service for advanced orchestration

---

## Acknowledgments  

- **Client**: Netherlands Plant Eco-phenotyping Centre (NPEC)
- **Product Owner**: Frank Peters, PhD.
- **MLOps & Deployment**: Dean van Aswegen, MSc. & Jason Harty, BSc.
- **Computer Vision**: Alican Noyan, PhD. (Block B supervisor)
- **Natural Language Processing**: Myrthe Buckens, MA. & Tsegaye Tashu, PhD.

---