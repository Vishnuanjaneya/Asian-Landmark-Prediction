# Asian Landmark Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter%20Notebooks-2.0.0-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/) 

This project aims to predict landmarks in Asian countries using TensorFlow Hub for model inference and PyQt for the user interface design.

## Introduction

Landmark prediction involves identifying specific points of interest or landmarks in images. This project focuses on predicting landmarks in various Asian countries, including iconic structures, temples, monuments, and natural landmarks.

## Requirements

| Dependency | Badge |
|------------|-------|
| Python Version | [![Python Version](https://img.shields.io/badge/Python-3.11.1-blue.svg)](https://www.python.org/downloads/release/python-3111/) |
| TensorFlow | [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) |
| OpenCV | [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/) |
| PyQt | [![PyQt](https://img.shields.io/badge/PyQt-5.x-blueviolet.svg)](https://riverbankcomputing.com/software/pyqt/intro) |



## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Vishnuanjaneya/Asian-Landmark-Prediction.git
    ```

2. Install Python 3.11.1 from [Python's official website](https://www.python.org/downloads/release/python-3111/).

3. Install dependencies using pip:

    ```bash
    pip install tensorflow opencv-python PyQt5
    ```

4. Optionally, you can install other dependencies specified in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## UI Design with PyQt Designer

1. Install PyQt5 Designer if you haven't already. It's usually installed along with PyQt5 package.

2. Design your UI using PyQt Designer. Save the UI file (usually with a `.ui` extension) in your project directory.

3. Convert the UI file to Python code using the `pyuic5` command:

    ```bash
    pyuic5 -x your_ui_file.ui -o ui_main_window.py
    ```

    Replace `your_ui_file.ui` with the name of your UI file.

## Usage

1. Run the application using Python:

    ```bash
    python main.py
    ```

2. The application will launch, allowing you to interact with the UI for Asian landmark prediction.

## Example

![Taj Mahal](taj.jpg)
*Image: Taj Mahal*

<img src="aadhi%20yogi.jpeg" alt="Aadhi Yogi" width="700" height="600">
*Image: Aadhi Yogi*

## Model

The model used for landmark prediction is provided by TensorFlow Hub. It offers a pre-trained model that can be easily integrated into the project.

## Contributing

Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- TensorFlow Team for providing TensorFlow Hub and its pre-trained models.
- PyQt Team for providing PyQt5 for UI development.

## References

- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [PyQt5 Documentation](https://doc.qt.io/qtforpython/index.html)
