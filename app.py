from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the text-generation pipeline
text_gen_pipeline = pipeline("text-generation", model="gpt2")

@app.route('/generate-text', methods=['POST'])
def generate_text():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        
        # Check if prompt is provided
        if 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        
        prompt = data['prompt']
        
        # Generate text
        result = text_gen_pipeline(prompt, max_length=50, num_return_sequences=1)
        
        # Return the generated text
        return jsonify({"generated_text": result[0]['generated_text']}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
