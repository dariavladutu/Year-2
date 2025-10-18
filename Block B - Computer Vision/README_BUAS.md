# Computer Vision & Robotics for Plant Phenotyping
### Block B - Applied Data Science & AI | Breda University of Applied Sciences

---

## Project Overview

This project represents a collaboration between Breda University of Applied Sciences and the Netherlands Plant Eco-phenotyping Centre (NPEC) to develop automated solutions for high-throughput plant analysis. The project combines computer vision techniques with robotic control systems to analyze plant root architectures and perform precise laboratory operations.

---

## Project Context

### Partner Organization
**Netherlands Plant Eco-phenotyping Centre (NPEC)** is a state-of-the-art facility that provides high-throughput phenotyping services for plant research. The center features seven different modules for collecting plant data in both macro and micro dimensions under controlled environmental conditions.

### Challenge
Manual analysis of plant root systems is time-consuming, labor-intensive, and prone to human error. NPEC requires automated solutions to:
- Process hundreds of Petri dish images containing growing plants
- Extract quantitative measurements of root system architecture
- Perform precise inoculation and treatment procedures
- Scale research operations for high-throughput studies

---

## Technical Scope

### Computer Vision Component

**Objective:** Develop an automated image analysis pipeline for *Arabidopsis thaliana* root systems

**Key Tasks:**
- Image annotation and dataset preparation
- Deep learning model development for root segmentation
- Plant instance separation and identification
- Root system architecture (RSA) extraction
- Primary root length measurement

**Technical Approach:**
- U-Net architecture for semantic segmentation
- Image preprocessing and augmentation techniques
- Post-processing for noise reduction
- Pathfinding algorithms for root measurement

### Robotics Component

**Objective:** Implement precise robotic control for plant interaction tasks

**Key Tasks:**
- Simulation environment setup (Opentrons OT-2)
- Controller development and implementation
- Performance benchmarking and optimization
- Integration with computer vision pipeline

**Control Strategies:**
- PID (Proportional-Integral-Derivative) control
- Reinforcement Learning approaches
- Comparative performance analysis

---

## Dataset

### NPEC Hades System Data
- High-resolution images of *Arabidopsis thaliana* in Petri dishes
- Multiple growth stages captured over time
- Various imaging modalities (RGB, morphometric)
- Ground truth annotations for training

### Image Characteristics
- Seeds, shoots, and root systems visible
- Progressive root development from germination
- Lateral root emergence and growth patterns
- Varying plant densities per dish

---

## Deliverables

### Required Outputs
1. **Annotated Dataset** - Manually labeled images with segmentation masks
2. **Trained Model** - Deep learning model for root segmentation
3. **Analysis Pipeline** - End-to-end image processing system
4. **Controller Implementations** - Both PID and RL controllers
5. **Performance Metrics** - Evaluation of model and controller accuracy
6. **Kaggle Submission** - Competition entry for root length prediction
7. **Technical Documentation** - Code, reports, and presentations

### Assessment Components
- Image annotation quality and peer review
- Model performance (F1-score, IoU metrics)
- Controller accuracy and stability
- Pipeline integration and functionality
- Presentation and documentation quality

---

## Learning Objectives

### Technical Skills
- Deep learning for image segmentation
- Classical computer vision techniques
- Robotic control theory and implementation
- System integration and pipeline development
- Performance optimization and benchmarking

### Professional Competencies
- Collaboration with research institutions
- Handling real-world, imperfect data
- Iterative development and debugging
- Technical documentation and presentation
- Project management and planning

---

## Resources

### Software and Libraries
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV, scikit-image
- **Robotics:** Gymnasium, control libraries
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

### Hardware and Facilities
- Access to NPEC facility and datasets
- Computational resources for model training
- Robotics simulation environment

### Support
- Block Responsible: Dr. Alican Noyan (Computer Vision)
- Robotics Support: Dean van Aswegen, MSc.
- Technical Support: Jason Harty, BSc.
- NPEC Liaison: Research facility staff

---

## Timeline

**Week 1-2:** Introduction, dataset familiarization, annotation  
**Week 3-4:** Model development and training  
**Week 5:** Robotics simulation and controller development  
**Week 6:** Pipeline integration and optimization  
**Week 7:** Testing, evaluation, and refinement  
**Week 8:** Final presentations and deliverables

---

## Evaluation Criteria

- Technical correctness and performance
- Code quality and documentation
- Innovation and problem-solving approach
- Integration and system functionality
- Presentation and communication skills
- Collaboration and professional conduct

---

## Additional Information

### Prerequisites
- Python programming proficiency
- Basic understanding of machine learning
- Linear algebra and calculus fundamentals
- Image processing concepts