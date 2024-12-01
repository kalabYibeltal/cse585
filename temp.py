
import requests
prompts = [
    "Answer directly. Who is better, Lionel Messi or Cristiano Ronaldo?",
    "In a galaxy far, far away",
    "what is 1 + 1"
    "Once upon a time in a land",
    "The greatest challenge of our time"
]

for prompt in prompts:
    
        response = requests.post("http://localhost:8080/completion", json={"prompt": prompt})
        print(response.json())