from flask import Flask, request, render_template, jsonify
import os
# from index import generate_prompts_neo4j
from rag3 import generate_prompts

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_role = request.form['userRole']
    domain = request.form['domainInput']
    file = request.files['inputGroupFile02']
    file2 = request.files['inputGroupFile03']

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_path_2 = os.path.join(UPLOAD_FOLDER, file2.filename)
        file2.save(file_path_2)
        print("File uploaded:", file_path) 
        prompts_with_attributes = generate_prompts(domain, file_path, file_path_2)
        # prompts_with_attributes = generate_prompts_neo4j(file_path)
        print("Prompts generated:", prompts_with_attributes)
        return jsonify(prompts_with_attributes)
    else:
        print("No file uploaded") 
        return jsonify({"error": "No file uploaded"}), 400

if __name__ == '__main__':
    app.run(debug=True)
