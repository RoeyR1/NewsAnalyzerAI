
# AI News Headline Analyzer

This project utilizes an AI model to classify news headlines into one of five binary categories and score headlines on four numerical scales. Aims to enhance understanding and analysis of media by providing insights into the nature and tone of news headlines.

---

## Features

- **Binary Categorization**: Headlines classified into one or more of the following categories:
  - Health
  - Technology
  - Environment
  - Finance
  - Politics

- **Scaled Scoring**: Assigns a score (0 to 10) for the following dimensions:
  - **Sentiment**: The overall emotional tone (positive, neutral, or negative).
  - **Intensity**: How strongly/intensely the headline conveys information.
  - **Urgency**: The immediacy or time-sensitivity of the headline.
  - **Controversy**: The potential for disagreement or divisiveness of the headline.

---

## Methodology

1. **Data Collection**:
   - Over 1.2 million news headlines were sourced using the Hugging Face library.
   - Data cleaning and preprocessing were performed with the Pandas library to prepare the dataset for training.

2. **Data Sampling**:
   - Due to computational and time constraints, the dataset was reduced to 125,000 cleaned and representative headlines.

3. **Model Training**:
   - Binary categorization and scaled scoring were trained simultaneously, ensuring efficiency and consistency.

---

## Performance Evaluation

- **Binary Categorization**:
  - Achieved an average **F1 Score** of **0.97**, with all scores exceeding **0.95**.
  
- **Scaled Scoring**:
  - Mean Squared Error (MSE) averages approximately **0.3**, with all values below **0.4** and many around **0.27**.


## Tools & Libraries

- **Data Handling**: Hugging Face, Pandas
- **Model Training**: TensorFlow/PyTorch
- **Evaluation Metrics**: F1 Score, Mean Squared Error (MSE)
  
---
## Testing the Program

1. **Install Dependencies**: Ensure all dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **Retrieve Model File**: The model file is tracked by Git LFS. To download it, first install and initialize Git LFS, then run:

   ```bash
   git lfs pull
   ```

4. **Run the Application**: Start the chatbot by executing:

   ```bash
   python app.py
   ```

6. **Input a Headline**: Once the chatbot starts, input a news headline to receive the analysis.
   
   ```plaintext
   Enter a news headline (or 'quit' to exit): 
   ```

---
