# PDPL Compliance Copilot

An AI-powered project designed to support privacy policy analysis and compliance assessment under Saudi Arabia's Personal Data Protection Law (PDPL).

## Project Overview

PDPL Compliance Copilot is the final project of the AI Engineering Bootcamp. It combines Machine Learning (ML), Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) to support privacy policy classification, legal information retrieval, and PDPL compliance analysis.

## Key Components

- **Machine Learning:** Classifies privacy policy clauses into predefined categories.
- **Legal RAG:** Retrieves relevant legal articles to support evidence-based answers to PDPL-related questions.
- **Compliance Gap Analysis:** Supports the identification of potential gaps between privacy policies and applicable PDPL requirements.

## Datasets and Legal Sources

### 1. SDAIA – Policies and Regulations
- **Source:** [National Data Governance Platform](https://dgp.sdaia.gov.sa/wps/portal/pdp/knowledgecenter/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8ziPR1dzTwMgw2MDMOcTA3MjH39TE29jY0MLIz1w9EUhIZZAhUEGvl6OXoaGwQY60cRo98AB3A0IKw_Ck0JpgvACvBYEZyap1-QGxphkGWiCAAUDrxA/dz/d5/L0lHSkovd0RNQUZrQUVnQSEhLzROVkUvYXI!/)
- **Purpose:** Official legal references for Saudi PDPL and its relevant regulations.

### 2. OPP-115 Dataset
- **Source:** [OPP-115 on Hugging Face](https://huggingface.co/datasets/alzoubi36/opp_115)
- **Purpose:** Supports privacy policy clause classification and multi-label machine learning experiments.

### 3. PDPL Legal Documents
- **Languages:** Arabic and English.
- **Format:** Six structured CSV files containing legal articles from the PDPL, Implementing Regulations, and Regulation on Personal Data Transfer Outside the Kingdom.
- **Purpose:** Used as the legal knowledge base for the RAG system.

## Technologies Used

- Python
- Machine Learning and NLP
- BM25, Dense Retrieval, and Hybrid Retrieval
- FAISS
- pandas
- Git and GitHub

## Project Structure

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

## Evaluation Results

Model performance and RAG evaluation results will be added after completing the final evaluation.

## Project Status

Under development.

## Acknowledgments

Developed as the final project of the AI Engineering Bootcamp.
