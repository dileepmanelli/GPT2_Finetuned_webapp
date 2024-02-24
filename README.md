## ML Web App with Fine-Tuned GPT-2 Model
Welcome to our ML Web App project! This application utilizes a fine-tuned GPT-2 language model to generate responses based on user input. Below, we provide detailed information on the project structure, implementation steps, and how we fine-tuned the GPT-2 model on custom data.

## Project Structure
```
.
├── app.py                 # Flask application file
├── fine_tuned_model/      # Directory containing the fine-tuned GPT-2 model
├── templates/             # Directory containing HTML templates
│   ├── index.html         # Main page template
│   └── result.html        # Result page template
└── requirements.txt       # List of Python dependencies
```
## Implementation Steps
Setting Up Flask: We initialized a Flask application in app.py to handle web requests and responses.

Fine-Tuning GPT-2 Model: We fine-tuned the GPT-2 language model on custom data related to our application domain. This involved:

Preprocessing the data to format it for GPT-2 training.
Fine-tuning the GPT-2 model using the transformers library from Hugging Face.
Saving the fine-tuned model to the fine_tuned_model/ directory.
Web Interface: We created two HTML templates, index.html and result.html, stored in the templates/ directory. These templates define the structure of the main page and the result page of the web app.

Flask Routes: We defined Flask routes in app.py to handle user requests:

The / route renders the main page where users can input text.
The /predict route processes user input, generates a response using the fine-tuned GPT-2 model, and renders the result page with the generated response.
Dependencies: We listed all Python dependencies required for the project in requirements.txt.

## Fine-Tuning GPT-2 Model
We fine-tuned the GPT-2 model on custom data to improve its performance for our specific use case. The fine-tuning process involved:

Selecting a pre-trained GPT-2 model suitable for our task.
Preprocessing our custom data to match the input format expected by GPT-2.
Training the model on the custom data while adjusting hyperparameters as needed.
Evaluating the fine-tuned model to ensure it meets performance requirements.
