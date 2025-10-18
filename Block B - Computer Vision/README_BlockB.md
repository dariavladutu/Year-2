# AI-Driven Plant Phenotyping: Computer Vision and Robotics
### Year 2 – Block B | Applied Data Science & AI | Breda University of Applied Sciences  
**Author:** Daria-Elena Vlăduțu

---

## Project Overview  
This project was developed in collaboration with the **Netherlands Plant Eco-phenotyping Centre (NPEC)** to address challenges in high-throughput plant analysis. The goal was to create an end-to-end automated pipeline that integrates computer vision for analyzing the root systems of *Arabidopsis thaliana* and robotics for precise plant inoculation. The solution focuses on replacing manual, time-consuming, and error-prone lab processes with a scalable, AI-driven system, thereby accelerating plant science research.

---

## Project Goals & Objectives  

### Main Goal
To design and implement an automated system capable of analyzing plant root imagery and controlling a robotic arm for precise interaction, streamlining the experimental workflow for NPEC.

### Key Objectives
- **Automated Image Analysis**: Develop a deep learning model to accurately segment plant root systems from high-resolution images of Petri dishes
- **Root System Measurement**: Extract key architectural traits, such as primary root length, from the segmented masks
- **Robotic Control**: Implement and compare controllers (PID and Reinforcement Learning) to guide a robotic arm to specific coordinates for tasks like inoculation
- **Pipeline Integration**: Combine the computer vision and robotics components into a single, functional pipeline

---

## Methodology & Pipeline  

### Computer Vision Component

#### Image Annotation & Dataset Preparation
- Manually annotated 15 images with 3 classes (shoot, seed, root) using LabKit to create ground truth segmentation masks
- Performed peer-review to ensure annotation quality
- Developed a robust preprocessing pipeline including:
  - Petri dish detection and cropping
  - Patchifying images into 256×256 segments for model training
  - Multiple dataset filtering strategies (simple, tuned, augmented, balanced)
- Tested different dataset preparation methods (all thoroughly documented under the **dataset_prep_for_training** folder, in the **task_4** notebook)

#### Root Segmentation Model
- **Architecture**: Custom U-Net convolutional neural network designed for biomedical image segmentation
- **Training**: Binary cross-entropy loss function
- **Performance**: Final model validation F1-score of **0.8374**
- **Post-processing**: Thresholding and area filtering (min_area = 150px) to clean raw model output and remove noise
- Trained the same model architecture with the different preprocessed datasets (all thoroughly documented under the **dataset_prep_for_training** folder, in the **task_4** notebook). All training iterations are thoroughly documented under the **model_training** folder.

#### Root System Architecture (RSA) Extraction
- Segmented individual plant instances from processed masks
- Skeletonized root structures to identify tip nodes
- Initially used scoring system based on path length, verticality, and centrality
- **Improvement**: Implemented A* search algorithm to find shortest path between highest and lowest tip nodes for more accurate primary root length measurement

### Robotics Component

#### Controller Development
Developed two controllers for the Opentrons OT-2 robot in a simulated environment:

**PID Controller**
- Tuned gains: KP=5.0, KI=0.5, KD=2.0
- Final positioning error: **0.592mm**
- Superior accuracy and stability for precision tasks

**Reinforcement Learning Agent**
- Trained through multiple iterations
- Final positioning error: 9.2mm
- Less suitable for high-precision requirements

#### Performance Benchmarking
Controllers evaluated based on:
- Positioning accuracy
- Path efficiency
- System stability
- Convergence time

---

## Key Findings & Results  

### Computer Vision Results
- U-Net model proved highly effective for segmenting complex, thin structures of plant roots
- Post-processing critical for refining predictions and removing salt-and-pepper noise
- A* search algorithm provided more robust method for calculating root length compared to heuristic-based scoring
- **Limitations**: Model struggled with very small, newly germinated roots and overlapping root systems

### Robotics Results
- PID controller demonstrated superior accuracy and stability for precise positioning tasks
- Sub-millimeter precision achieved (0.592mm error)
- Reinforcement Learning agent successfully trained but showed larger final error
- PID controller selected as optimal solution for client requirements

---

## Deliverables  

- **Image Annotations**: 15 high-quality segmentation masks for the Y2B_24 dataset
- **Computer Vision Pipeline**: Python scripts and notebooks for:
  - Petri dish detection and extraction
  - Plant instance segmentation
  - Root system architecture extraction
- **Trained Deep Learning Model**: Final U-Net model for root segmentation
- **Kaggle Competition Submission**: Predictions of primary root lengths
- **Robotic Controllers**: Implementations of both PID and Reinforcement Learning controllers
- **Final Client Presentation**: Comprehensive presentation of methodology, results, and limitations

---

## Skills Demonstrated  

### Technical Skills
- **Computer Vision**: Image segmentation, object detection, preprocessing & post-processing
- **Deep Learning**: U-Net architecture, TensorFlow, Keras
- **Robotics**: PID control, Reinforcement Learning, simulation environments (Gymnasium)
- **Data Science**: Experimental design, data annotation, performance metrics (F1-Score, SMAPE)
- **Algorithm Design**: A* search, pathfinding algorithms
- **Software Engineering**: Python, OpenCV, scikit-image, Git & GitHub

### Professional Skills
- Problem solving through iterative development
- Error analysis and pipeline optimization
- Technical documentation and presentation
- Collaboration with research institution

---

## Tools & Technologies  

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, scikit-image, Pandas, NumPy, Matplotlib, Optuna
- **Annotation Tool**: LabKit (ImageJ)
- **Robotics Simulation**: Gymnasium
- **Version Control**: Git & GitHub

---

## Repository Structure  

```
Block B - Computer Vision/
├── computer_vision/
│   ├── dataset_prep_for_training/
│   ├── final_presentation/
│   ├── image_annotation_and_review/
│   ├── individual_root_segmentation/
│   ├── kaggle_competition/
│   ├── model_training/
│   ├── petri_dish_detection_and_extraction/
│   ├── PID_controller/
│   ├── plant_instance_segmentation/
│   ├── RSA_extraction/
│   └── simulation_environment_setup/
├── robotics/
|    └── # Contents for robotics part (e.g., RL controller, etc.)
├── final_presentation/
│   └── AI and the Arabidopsis thaliana.pdf 
└──
```

---

## Impact & Applications  

This project demonstrates practical applications of AI in agricultural technology:
- **Efficiency**: 10× faster analysis compared to manual methods
- **Accuracy**: Consistent, reproducible measurements
- **Scalability**: Processes hundreds of samples automatically
- **Research Acceleration**: Enables high-throughput phenotyping studies

---

## Future Work  

- Extend model to support multiple plant species
- Incorporate 3D analysis for root volume measurements
- Optimize pipeline for real-time processing
- Implement more sophisticated RL algorithms
- Develop cloud-based deployment for remote analysis

---

## Acknowledgments  

- **Client**: Netherlands Plant Eco-phenotyping Centre (NPEC)
- **Block Responsible**: Dr. Alican Noyan
- **Robotics & RL Support**: Dean van Aswegen, MSc. & Jason Harty, BSc.

---