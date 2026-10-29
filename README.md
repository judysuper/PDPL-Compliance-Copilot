# PDPL Compliance Copilot

An AI-driven compliance support system that leverages **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Retrieval-Augmented Generation (RAG)** to facilitate privacy policy analysis against Saudi Arabia's Personal Data Protection Law (PDPL).

## 1. Project Overview

PDPL Compliance Copilot is the final project of the **AI Engineering Bootcamp**. It aims to support privacy compliance assessment by integrating legal information retrieval, multi-label privacy policy classification, and compliance gap analysis.

The system combines structured legal documents with machine learning and information retrieval techniques to identify relevant legal provisions and support privacy policy analysis.

## 2. System Architecture

The project consists of three main components:

- **Privacy Policy Classification:** Applies supervised machine learning and multi-label text classification to categorize privacy policy clauses according to predefined privacy practices.
- **Legal Information Retrieval (RAG):** Uses lexical and dense retrieval techniques to retrieve relevant legal provisions and support responses to PDPL-related queries.
- **Compliance Gap Analysis:** Aims to identify potential discrepancies between privacy policy statements and applicable legal requirements.

## 3. Datasets and Data Sources

### SDAIA — Official Legal Sources
**Source:** [National Data Governance Platform](https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/!ut/p/z1/04_Sj9CPykss0xPLMnMz0vMAfIjo8ziPR1dzTwMgw2MDMOcTA3MjH39TE29jY0MLIz1w9EUhIZZAhUEGvl6OXoaGwQY60cRo98AB3A0IKw_Ck0JpgvACvBYEZyap1-QGxphkGWiCAAUDrxA/dz/d5/L0lHSkovd0RNQUZrQUVnQSEhLzROVkUvYXI!/)

**Purpose:** Provides official legal references for Saudi Arabia's Personal Data Protection Law and its associated regulations.

### OPP-115 Dataset
**Source:** [OPP-115 on Hugging Face](https://huggingface.co/datasets/alzoubi36/opp_115)

**Purpose:** Supports the development of multi-label machine learning models for privacy policy clause classification.

### PDPL Legal Documents

The legal knowledge base consists of six structured CSV files containing Arabic and English article-level text from:

- Personal Data Protection Law (PDPL)
- Implementing Regulations
- Regulation on Personal Data Transfer Outside the Kingdom

**Preprocessing:** The documents were converted from PDF to text, cleaned, and segmented into article-level records.

**Purpose:** These structured documents support legal information retrieval and the development of evidence-grounded responses.

### Team-built Gold Q&A Set

**Contents:** At least 60 questions, including at least 20 Arabic questions and 10 unanswerable questions, each with gold article number(s) and a reference answer.

**Purpose:** Evaluates retrieval relevance, answer correctness, legal citation accuracy, and the system's ability to abstain when the available legal corpus cannot answer a question.

### Team-Annotated Local Privacy Policies

**Contents:** At least 10 publicly available privacy policies from apps or websites operating in Saudi Arabia, annotated by the team against the PDPL compliance checklist.

**Purpose:** Supports the assessment of privacy policy practices against Saudi PDPL requirements and helps identify potential compliance gaps.



## 4. Methodology

The project follows two complementary workflows:

**Machine Learning Workflow**
1. Dataset inspection and quality assessment.
2. Text preprocessing and multi-label target preparation.
3. Baseline model development.
4. Transformer-based model training and evaluation.
5. Model performance comparison.

**Legal Retrieval Workflow**
1. Legal document extraction and text cleaning.
2. Article-level segmentation and structured storage.
3. Lexical retrieval using BM25.
4. Dense retrieval using semantic text representations.
5. Hybrid retrieval to combine lexical and semantic results.
6. Retrieval evaluation and evidence verification.

## 5. Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, Transformer-based models
- **Natural Language Processing:** Text preprocessing and multilingual language models
- **Information Retrieval:** BM25, Dense Retrieval, Hybrid Retrieval
- **Vector Search:** FAISS
- **Data Processing:** pandas
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git and GitHub

## 6. Project Structure

```text
PDPL-Compliance-Copilot/
├── app/
├── data/
├── docs/
├── notebooks/
├── reports/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## 7. Evaluation and Results

Classification and retrieval performance will be documented following the final evaluation.

Evaluation will focus on:
- Multi-label classification performance.
- Comparison between baseline and transformer-based models.
- Retrieval relevance and ranking quality.
- Legal citation accuracy and supporting evidence.

Quantitative results will be added once the final experiments and validation are complete.

## 8. Installation and Usage

Installation and execution instructions will be provided after the project dependencies and application workflow are finalized.

## 9. Project Status

**Status:** Under development.

The project is undergoing model evaluation, retrieval validation, and documentation.

## 10. Acknowledgments

Developed as the final project of the **AI Engineering Bootcamp**.
