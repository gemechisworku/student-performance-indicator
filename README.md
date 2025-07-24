## Student Performance Indicator - End-to-End ML Project
This project is a comprehensive end-to-end machine learning application designed to predict student performance based on various features.
It includes data preprocessing, model training, evaluation, and deployment using Flask and AWS Elastic Beanstalk.
## Features
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Model Training**: Supports multiple regression models with hyperparameter tuning using GridSearchCV.
- **Model Evaluation**: Provides detailed evaluation metrics including R-squared, Mean Absolute Error, and Mean Squared Error.
- **Deployment**: Easily deployable on AWS Elastic Beanstalk with a simple Flask application.
- **User Interface**: A user-friendly web interface for inputting student data and displaying predictions.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gemechisworku/student-performance-indicator.git
    cd student-performance-indicator
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
### Usage
1. Start the Flask application:
   ```bash
   python application.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Input the student data in the form and submit to get predictions.
4. The application will display the predicted student performance along with evaluation metrics.
5. For deployment, follow the instructions in the `README.md` file for AWS Elastic Beanstalk.
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE)) file for details
