# Cyber Attack Detection using A.I.

This repository contains a state-of-the-art application for detecting potential cyber threats. Leveraging advanced machine learning techniques and trained on a comprehensive dataset sourced from MalwareBazaar, the tool identifies malware from binary and portable executable files, offering insights into 15 of the most prevalent malware classes.

## Features
- Built with Streamlit for an intuitive user interface.
- Uses TensorFlow for deep learning-based malware detection.
- Classifies uploaded files against 15 malware classes.
- Provides confidence scores for detected threats.
- Converts binary files to RGB images for deep learning-based classification.

## Malware Classes
The model can identify the following malware classes:
- AgentTesla
- Amadey
- AsyncRAT
- Emotet
- FormBook
- GuLoader
- Loki
- Mirai
- NetSupportRAT
- NjRat
- Non-Malicious
- QBot
- RedLineStealer
- Remcos
- Vidar

## Setup & Installation

1. Clone this repository:
```bash
git clone https://github.com/zahidhasanshuvo/cyber-attack-detection.git
```

2. Navigate to the project directory:
```bash
cd path-to-directory
```

3. Install the necessary libraries:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## How to Use
1. Launch the application.
2. Upload the binary or portable executable file you want to inspect.
3. Click on "Scan Now" and wait for the results.
4. The application will classify the file and provide a confidence score.

## Contribution
Feel free to fork the repository and submit pull requests. For major changes, open an issue first to discuss the proposed changes.

## Acknowledgments
- Dataset and insights sourced from [MalwareBazaar](https://bazaar.abuse.ch/).
- Dataset samples collected from [Vx-Underground](https://www.vx-underground.org/).
- Special thanks to the open-source community for various resources and tools that aided this project.

## Footer
Developed with ❤️ by [Zahid](https://github.com/zahidhasanshuvo).

---
This README provides a structured overview of the project, guides for setup, and acknowledgment.