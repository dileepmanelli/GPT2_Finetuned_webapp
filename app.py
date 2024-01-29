import torch
from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__, template_folder='templates')

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_model'  # Adjust the path if needed
fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_response(question, max_length=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize the input question
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

    # Generate a response with attention mask
    with torch.no_grad():
        # Create attention mask tensor
        attention_mask = torch.ones(input_ids.shape, device=device)

        # Generate a response with attention mask
        output = fine_tuned_model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)

    # Decode and return the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    response = generate_response(user_input)

    return render_template('result.html', input_text=user_input, response_text=response)

if __name__ == '__main__':
    app.run(debug=True)

