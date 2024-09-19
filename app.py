from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from flask import Flask,render_template,request
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
import re
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are my Personal assistant."),
        ("human","Question: {question}")
    ]
)

model = Ollama(model="llama3")

parser = StrOutputParser()

chain = prompt | model | parser

@app.route("/",methods=["POST","GET"])
def main():
    query_input = None
    output = None

    if request.method == "POST":
        query_input = request.form.get("query-input")
        if query_input:
            try:
                response = chain.invoke({"question":query_input})
                output = format_output(response)
            except Exception as e:
                logging.error("Error has occured while execution.")
                return f"There is an error which has occured: {e}"
    return render_template("index.html",query_input=query_input,output=output)

if __name__ == "__main__":
    app.run(debug=True)
    


