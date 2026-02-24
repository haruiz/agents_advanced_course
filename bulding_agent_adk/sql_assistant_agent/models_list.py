from google import genai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = genai.Client()

print("List of models that support generateContent:\n")
for m in client.models.list():
    # for action in m.supported_actions:
    #     if action == "generateContent":
    print(m.name)

print("List of models that support bidiGenerateContent:\n")
for m in client.models.list():
    if "bidiGenerateContent" in m.supported_actions and "generateContent" in m.supported_actions:
        print(m.name)

