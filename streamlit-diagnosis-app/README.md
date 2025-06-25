# Streamlit Disease Diagnosis Application

This project is a Streamlit application designed to assist users in diagnosing diseases based on their symptoms and vital signs. The application allows users to input various symptoms and health metrics, processes the data, and provides a diagnosis along with a recommended treatment plan.

## Project Structure

```
streamlit-diagnosis-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── model
│   │   ├── __init__.py       # Initializes the model package
│   │   └── predictor.py       # Contains logic for making predictions
│   ├── utils
│   │   ├── __init__.py       # Initializes the utils package
│   │   └── preprocessing.py    # Functions for preprocessing input data
│   └── data
│       └── case_base.csv      # Historical case data for training and predictions
├── requirements.txt           # Lists project dependencies
└── README.md                  # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamlit-diagnosis-app
   ```

2. **Install the required packages**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   Execute the following command to start the application:
   ```bash
   streamlit run src/app.py
   ```

## Usage Guidelines

- Upon launching the application, users will be prompted to enter their symptoms, blood pressure, heart rate, body temperature, and oxygen saturation levels.
- After submitting the input, the application will process the data and display the predicted diagnosis and treatment plan based on the input provided.

## Contributing

Contributions to enhance the application are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.