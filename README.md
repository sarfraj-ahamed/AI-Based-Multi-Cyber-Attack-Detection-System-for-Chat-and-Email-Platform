\# AI-Based Multi Cyber Attack Detection System for Chat and Email Platform



A deep learning-based multi-modal cyber attack detection system for chat and email platforms, integrating NLP, steganalysis, and malware analysis using hybrid architectures with automated workflows.



\## Overview



This project presents a unified cybersecurity framework to detect major threat classes in digital communication systems:



\- phishing attacks in emails and chat messages

\- malware embedded in attachments

\- suspicious and malicious content handled through modular AI pipelines



The system combines deep learning models, API-based deployment, and workflow automation for real-time threat detection and alerting.



\## Key Features



\- phishing detection using NLP and URL-aware analysis

\- malware attachment analysis using deep learning

\- modular project structure for easy extension

\- FastAPI-based APIs for detection services

\- real-time workflow integration readiness



\## Modules



\### Phishing Detection

This module analyzes suspicious text and URLs from emails or chat messages.



\### Malware Detection

This module scans suspicious attachments and classifies them using deep learning pipelines.



\## Project Structure



```text

assets/

malware/

&#x20; api/

&#x20; models/

&#x20; scripts/

&#x20; results/

phishing/

&#x20; api/

&#x20; models/

&#x20; scripts/

&#x20; results/

README.md

requirements.txt

.gitignore

LICENSE

CITATION.cff

Tech Stack
Python
FastAPI
TensorFlow
PyTorch
Transformers
Streamlit
NumPy
Pandas
OpenCV
Setup
git clone https://github.com/sarfraj-ahamed/AI-Based-Multi-Cyber-Attack-Detection-System-for-Chat-and-Email-Platform.git
cd AI-Based-Multi-Cyber-Attack-Detection-System-for-Chat-and-Email-Platform
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Results

Add your result screenshots, confusion matrices, ROC curves, and evaluation outputs inside the assets/, malware/results/, and phishing/results/ folders.

Author

Sarfraj Ahamed S
GitHub: https://github.com/sarfraj-ahamed

LinkedIn: https://www.linkedin.com/in/sarfraj-s-ahamed

License

This project is licensed under the MIT License.