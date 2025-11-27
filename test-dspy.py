import dspy
import os
import litellm
from pathlib import Path


api_base = "http://localhost:11434"

apikey = os.getenv('DEEPSEEK_API_KEY')

lm  = dspy.LM('deepseek/deepseek-chat', api_key=apikey)

dspy.configure(lm=lm)

class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()
    
    
predict = dspy.Predict(BasicQA)

history = dspy.History(messages=[])

while True:
    question =  input("Enter your question, if you want to exit type 'finish': ")
    if question.lower() == 'finish':
        break
    output = predict(question=question, history=history)
    print(f"User Question: {question}\nSystem Answer: {output.answer}")
    history.messages.append({"questions": question, **output})
    
    
dspy.inspect_history()