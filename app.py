from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF library for PDF parsing
import time
from crewai import Agent, Crew, Task, Process
import openai

# Set environment variables for API access
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3-70b-8192"
os.environ["OPENAI_API_KEY"] = "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"

# Define the Explanation Generator Agent
explanation_generator = Agent(
    role='explanation generator',
    goal='to provide concise explanations for student questions in a formatted way',
    backstory='you are an AI teacher helping to explain topics to students',
    verbose=True,
    allow_delegation=False
)

app = Flask(__name__)

# Initialize the chat history and PDF content
chat_history = []
pdf_content = ""

def truncate_input(input_text, max_length=4000):
    return input_text[:max_length]

def format_response(response_text):
    # Custom formatting logic
    formatted_response = response_text.replace("**", "<strong>").replace("*", "<em>").replace("\n", "<br>")
    return formatted_response

def generate_explanation(question):
    global pdf_content
    context = f"{pdf_content}\n\n{chat_history}"
    truncated_context = truncate_input(context, max_length=4000)

    generate_explanation_task = Task(
        description=f"generate a concise and formatted explanation for the following question: {question} with the context: {truncated_context}",
        agent=explanation_generator,
        expected_output="concise and formatted explanation of the topic"
    )

    crew = Crew(
        agents=[explanation_generator],
        tasks=[generate_explanation_task],
        verbose=2,
        process=Process.sequential
    )

    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            output = crew.kickoff()
            return format_response(output)
        except openai.RateLimitError as e:
            retry_count += 1
            wait_time = 60 * (2 ** retry_count)  # Exponential backoff
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return "Sorry, I am currently experiencing high demand. Please try again later."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    bot_response = generate_explanation(question)
    chat_history.append({"user": question, "bot": bot_response})
    return jsonify({"response": bot_response})
@app.route('/upload', methods=['POST'])
def upload():
    global pdf_content
    file = request.files['file']
    if file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        pdf_content = ""
        try:
            with open(file_path, 'rb') as f:
                doc = fitz.open(file_path)
                for page in doc:
                    pdf_content += page.get_text()
                doc.close()  # Close the PDF document after reading
            return jsonify({"message": "PDF content successfully parsed."})
        except Exception as e:
            return jsonify({"message": f"Error parsing PDF: {e}"}), 500
        finally:
            os.remove(file_path)  # Remove the file after processing
    return jsonify({"message": "No file uploaded."}), 400


if __name__ == '__main__':
    app.run(debug=True)
