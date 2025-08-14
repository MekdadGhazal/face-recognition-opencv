# Real-Time Face Recognition with OpenCV

[![Python Version][python-badge]][python-link]
[![OpenCV Version][opencv-badge]][opencv-link]
[![License: MIT][license-badge]][license-link]
[![Repo Status: Active][status-badge]][status-link]

<p align="center">
  <img src="https://i.imgur.com/gYdQB4k.gif" alt="Face Recognition Demo" width="700"/>
</p>

A comprehensive desktop application built in Python that uses the **OpenCV** library and the **LBPH (Local Binary Patterns Histograms )** algorithm to create a real-time face recognition system. The application allows users to register new faces, train the model on all registered individuals, and then recognize them through a live webcam feed.

---

## 🌟 Key Features

*   **Real-Time Recognition:** Identifies faces directly from the live camera stream.
*   **New User Registration:** An easy-to-use interface for collecting and labeling new face data.
*   **Comprehensive Model Training:** Trains the model on **all** registered users to ensure high accuracy.
*   **Clean Code Architecture:** Built with Object-Oriented Programming (OOP) principles for modularity and easy maintenance.
*   **Automatic Saving:** Automatically saves the updated model (`model.yml`) and labels file (`labels.json`).
*   **Simple User Interface:** Fully controlled via keyboard shortcuts.

---

## 🛠️ Tech Stack

*   **Python 3.x**
*   **OpenCV (`opencv-python`)**: For image/video processing and face detection.
*   **NumPy**: For numerical operations on arrays.
*   **JSON**: For storing and managing user labels.

---

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

*   Ensure you have **Python 3.8** or newer installed on your system.
*   Ensure you have **Git** installed on your system.

### 2. Clone the Repository

Open your terminal and clone this repository to your local machine:
```bash
git clone https://github.com/MekdadGhazal/face-recognition-opencv.git
cd face-recognition-opencv
```

### 3. Create a Virtual Environment and Install Dependencies

It is a best practice to use a virtual environment to isolate project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```
*(Note: You will need to create a `requirements.txt` file containing `opencv-python` and `numpy` )*

### 4. Run the Application

After activating the environment and installing the dependencies, run the main script:
```bash
python run.py
```

---

## 📖 Usage Guide

When you run the application, the webcam feed window will appear. You can control the application using the following keys:

*   **`Y` - Register a New Face:**
    1.  Press `Y`.
    2.  A prompt will appear in the terminal asking you to enter a name.
    3.  Type the name and press `Enter`.
    4.  The application will automatically start capturing 50 images of your face. Try to move your head slightly to capture different angles.
    5.  Once finished, the application will automatically retrain the model on all registered faces.

*   **`N` - Toggle Recognition Mode:**
    *   Press `N` to activate recognition mode. The application will start drawing boxes around faces and identifying them.
    *   Press `N` again to deactivate recognition mode and return to the normal feed.

*   **`S` - Take a Screenshot:**
    *   Press `S` to save the current frame as an image in the `data/raw` directory.

*   **`X` - Exit the Application:**
    *   Press `X` to safely close the program.

---

## 📁 Project Structure

```
face-recognition-opencv/
│
├── .gitignore          # Specifies intentionally untracked files to ignore
├── data/               # Directory for storing data
│   ├── faces/          # Stores collected face images (one folder per user)
│   └── raw/            # Stores manual screenshots
│
├── labels.json         # Stores user names and their corresponding numeric labels
├── model.yml           # The trained recognizer model, created automatically
├── run.py              # The main application script
└── README.md           # This file
```

---

## 🤝 Contributing

Contributions are always welcome! If you have a suggestion or a fix, feel free to fork the repository and create a pull request.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

<!-- Badges -->
[python-badge]: https://img.shields.io/badge/Python-3.8%2B-blue.svg
[python-link]: https://www.python.org/
[opencv-badge]: https://img.shields.io/badge/OpenCV-4.x-green.svg
[opencv-link]: https://opencv.org/
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-link]: https://opensource.org/licenses/MIT
[status-badge]: https://img.shields.io/badge/Repo%20Status-Active-brightgreen
[status-link]: #

