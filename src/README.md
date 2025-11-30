# `/src` Directory - Medical System Components

This directory contains the core modules of the Chronic Kidney Disease (CKD) prediction and management system. It includes data extraction, machine learning models, risk assessment algorithms, and a medical simulation interface.

## Directory Structure

### 1. **`algorithm_module/`** - Risk Assessment API
Flask-based REST API for assessing chronic kidney disease risk using trained models.

**Key Files:**
- `main.py` - Flask application with `/api-asses-risk` endpoint
- `environment.py` - Environment configuration and dependencies
- `algorithm.py` - Algorithm

**Usage:**
```bash
cd algorithm_module
python main.py
```
Exposes endpoint: `POST /api-asses-risk` to process patient risk assessments.

---

### 2. **`extractor_module/`** - PDF/Data Extraction
Extracts patient data from PDF documents and clinical records.

**Key Files:**
- `main.py` - Flask application with extraction endpoints
- `extractor.py` - Core extraction logic for patient data
- `environment.py` - Configuration and processing
- `tmp/` - Temporary extracted data storage

**Usage:**
```bash
cd extractor_module
python main.py
```
Exposes endpoints to extract patient information from documents and structured data.

---

### 3. **`models/`** - Machine Learning Models
Contains training and testing scripts for CKD prediction models.

**Key Files:**
- `train.py` - Training pipeline for Logistic Regression and XGBoost models
- `test.py` - Model evaluation and testing utilities
- `main.py` - Main script for model training and evaluation
- `patient_data_extracted.json` - Sample extracted patient data
- `prediction_result.json` - Sample prediction outputs
- `results/` - Training metrics and summaries
- `weights/` - Saved model weights

**Usage:**
```bash
cd models
python main.py            
```

---

### 4. **`response_module/`** - Response Handler
Processes and formats responses from the prediction system.

**Key Files:**
- `main.py` - Flask application for response handling
- `environment.py` - Configuration setup
- `reports/` - Generated patient reports

**Usage:**
```bash
cd response_module
python main.py
```

---

### 5. **`medical_system_simulation/`** - Patient Management Interface
Flask-based web application simulating a medical management system for patient records and testing.

**Key Files:**
- `main.py` - Flask web application
- `environment.py` - Configuration
- `static/` - CSS/JS assets
- `templates/` - HTML templates
  - `index.html` - Main dashboard
  - `patient_record.html` - Patient detail view
- `logs/` - Application logs

**Usage:**
```bash
cd medical_system_simulation
python main.py
```
Access web interface at `http://localhost:8000` (or configured port).

---

### 6. **`infrastructure/`** - Infrastructure as Code
Terraform configurations for AWS deployment.

**Key Files:**
- `main.tf` - Main infrastructure definition
- `variables.tf` - Variable declarations
- `outputs.tf` - Output definitions
- `vpc.tf` - Virtual Private Cloud configuration
- `alb.tf` - Application Load Balancer
- `asg.tf` - Auto Scaling Group
- `iam.tf` - Identity and Access Management
- `security.tf` - Security groups and policies
- `backend.tf` - Terraform state backend
- `Jenkinsfile` - CI/CD pipeline definition

**Usage:**
```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```

---

##  Setup and Installation

### Prerequisites
- Python 3.12.1
- pip (Python package manager)
- Terraform (for infrastructure deployment)

### Install Dependencies
```bash
cd src
pip install -r requirements.txt
```

### Key Dependencies
- **Flask** - Web framework for APIs
- **scikit-learn** - Machine learning models
- **XGBoost** - Gradient boosting models
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **Google GenAI** - AI integration
- **ReportLab** - Report generation

---

## Quick Start

### Option 1: Run Individual Modules

**Start the Algorithm Service (Risk Assessment):**
```bash
cd src/algorithm_module
python main.py
```

**Start the Data Extractor Service:**
```bash
cd src/extractor_module
python main.py
```

**Train ML Models:**
```bash
cd src/models
python main.py --test
```

**Start the Medical Simulation Interface:**
```bash
cd src/medical_system_simulation
python main.py
```

### Option 2: Deploy to AWS
```bash
cd src/infrastructure
terraform init
terraform plan
terraform apply
```

---

## Workflow Overview

1. **Data Extraction** (`extractor_module`) → Extract patient data from documents
2. **Model Training** (`models`) → Train prediction models on historical data
3. **Risk Assessment** (`algorithm_module`) → Evaluate patient risk using trained models
4. **Response Processing** (`response_module`) → Format and deliver results
5. **Patient Management** (`medical_system_simulation`) → View/manage patient records

---

##  Configuration

Each module has an `environment.py` file for configuration. Environment variables can be set in `.env` files located in each module's `env/` directory.

---

##  Additional Resources

- Model training outputs: `models/results/`
- Saved model weights: `models/weights/` and `algorithm_module/weights/`
- Sample data: `data/Chronic_Kidney_Dsease_data.csv`
- Patient data: `data/patients/`

---

##  Troubleshooting

- **Module imports fail**: Ensure you're in the correct directory when running scripts
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Port already in use**: Change the port in the respective `environment.py` file
- **Terraform errors**: Ensure AWS credentials are configured (`aws configure`)
