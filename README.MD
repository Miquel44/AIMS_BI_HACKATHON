# AIMS: Artificial Intelligence for Medical Solutions

A project developed during the **Boehringer Ingelheim Hackathon 2025** to support the early detection of Chronic Kidney Disease (CKD).

DISCLAIMER: This is the solution for the 7th edition of the Boehringer Hackathon in the next link:https://www.boehringer-ingelheim.com/es/comprometidos-por-el-futuro/
We chose the challenge 2, described below
## 💡 Why this project? - Challenge 2

Chronic Kidney Disease (CKD) affects millions of people worldwide and is often detected too late. We believe technology can make a difference by enabling earlier detection and better monitoring.

## 🎯 What does it do?

It’s a complete system with several steps. First, it extracts records from patient histories. Then, using an LLM, it reads this data and extracts relevant parameters. Afterwards, with our predictive model, it estimates the risk of CKD and provides actionable insights for healthcare professionals.

## 🚀 Key Features

- Analysis of multiple clinical parameters
- Intuitive process for healthcare professionals
- Clear and actionable results
- Easy integration with existing medical systems

## Architecture Design

![Modular Architecture](docsiagrams/Arquitectura de Aplicación.png)

### Cloud Infrastructure (AWS)

![AWS Infrastructure](docsiagrams/Arquitectura de Sistema.drawio.png)

- **High availability**: Deployment across multiple availability zones
- **Auto scaling**: Auto Scaling groups that adjust to demand
- **Security**: Security groups and isolated VPCs
- **Load balancing**: Efficient traffic distribution

### Processing Flow

![Flow Diagram](docsiagrams/Diagrama de Actividad.drawio.png)

---

## 💼 Business Model: 3-Step Consulting

### 1️⃣ Assessment
- Analysis of the existing medical system
- Identification of integration points
- Evaluation of specific requirements
- Resource estimation

### 2️⃣ Implementation and Integration
- Development of custom modules for data extraction and analysis
- Secure integration with the existing Electronic Health Record (EHR) system
- Rigorous testing to ensure data accuracy and system reliability

### 3️⃣ Support and Maintenance
- Continuous system monitoring to ensure optimal performance
- Regular updates to the predictive model and software components
- Ongoing technical support for healthcare professionals

## ✨ Key Advantages

- **Modular**: Each component is independent and reusable
- **Scalable**: Grows according to the medical center’s needs
- **Interoperable**: Integrates with existing systems without replacing them
- **Maintainable**: Updates without service interruptions
- **Secure**: Complies with medical privacy standards

---

## 🛠️ Technologies Used

### Backend and Core
- **Python 3.x** - Main language of the project
- **Flask 3.1.2** - Web framework for the API
- **Google Generative AI (Gemini) 1.52.0** - LLM for extracting medical data

### Machine Learning and Data Analysis
- **scikit-learn 1.7.1** - Predictive CKD model
- **XGBoost 2.0.3** - Machine learning algorithms
- **pandas 2.3.3** - Data manipulation
- **numpy 2.3.5** - Numerical computing
- **scipy 1.16.3** - Scientific analysis

### Visualization
- **matplotlib 3.10.6** - Charts and visualizations
- **seaborn 0.13.2** - Advanced statistical visualizations

### User Interface
- **PyQt6 6.9.1** - Desktop graphical interface

### Report Generation
- **ReportLab 4.4.5** - PDF report generation

### Utilities
- **requests 2.32.5** - HTTP client
- **BeautifulSoup4 4.14.2** - HTML/XML parsing
- **pydantic 2.12.5** - Data validation

## 👥 Team

Developed by a team committed to improving healthcare through technological innovation.
- Miquel González 
- Joel Salvador Tapia
- Sergio Martínez Pérez

---

**Note**: This project is a support tool and does not replace professional medical judgment.
