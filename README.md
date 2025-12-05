Intelligent Analysis of Nessus Vulnerabilities

A Streamlit application for automated vulnerability clustering and AI-powered interpretation.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/Ollama-Mistral-green" /> <img src="https://img.shields.io/badge/Clustering-KMeans%20%7C%20DBSCAN-orange" /> <img src="https://img.shields.io/badge/Database-ChromaDB-purple" /> </p>
Table of Contents

Features

Installation

Build the Vector Index

Run the Application

Project Structure

Technologies Used

Purpose

Screenshots (Optional)

Features

Dimensionality reduction with PCA

Clustering using KMeans and DBSCAN

Vector knowledge base using ChromaDB

Local AI processing via Ollama (Mistral)

Interactive Streamlit interface

Automated vulnerability interpretation and recommendations

Installation
pip install -r requirements.txt

Build the Vector Index
python app/build_index.py

Run the Application
streamlit run app/app.py

Project Structure
.
├── app/
│   ├── app.py               # Main Streamlit application
│   ├── build_index.py       # Script to generate vector index
│   ├── utils/               # Helper functions and modules
├── data/                    # Report samples and vector data
├── requirements.txt         # Dependencies

Technologies Used
Category	Tools & Libraries
AI / LLM	Ollama (Mistral), ChromaDB
Machine Learning	PCA, KMeans, DBSCAN
Application	Streamlit
Data Processing	Pandas, NumPy
Security Input	Nessus CSV Reports
Purpose

This project is intended for cybersecurity analysts, students, and researchers who need faster and more structured insights into Nessus vulnerability reports.
It provides clustering, visualization, and AI-assisted interpretation to improve vulnerability management and prioritization.

