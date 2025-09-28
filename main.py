from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import requests
import json
import subprocess

load_dotenv()

client = OpenAI()

# ---------------------------
# Tool 1 -> get_weather
# ---------------------------
def get_weather(params):
    """
    params: { "city": "London" }
    """
    city = params.get("city")
    if not city:
        return "Missing required key: 'city'"

    url = f"https://wttr.in/{city}?format=%C+%t"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.RequestException as e:
        return f"Something went wrong: {e}"


# ---------------------------
# Tool 2 -> run_command
# ---------------------------
def run_command(params):
    """
    params: { "cmd": "ls -l" }
    """
    cmd = params.get("cmd")
    if not cmd:
        return "Missing required key: 'cmd'"

    try:
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if completed.returncode == 0:
            return completed.stdout.strip() if completed.stdout else "Command executed successfully."
        else:
            return completed.stderr.strip()
    except Exception as e:
        return f"Failed to execute command: {e}"


# ---------------------------
# Tool 3 -> write_to_file
# ---------------------------
def write_to_file(params):
    """
    params: { "filename": "output.txt", "content": "Hello world" }
    """
    filename = params.get("filename")
    content = params.get("content", "")

    if not filename:
        return "Missing required key: 'filename'"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{filename}' created successfully."
    except Exception as e:
        return f"Failed to write file: {e}"


# ---------------------------
# Tool 4 -> read_file
# ---------------------------
def read_file(params):
    """
    params: { "filename": "output.txt" }
    """
    filename = params.get("filename")

    if not filename:
        return "Missing required key: 'filename'"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Failed to read file: {e}"


# ---------------------------
# Tool Registry
# ---------------------------
available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
    "write_to_file": write_to_file,
    "read_file": read_file,
}


# ---------------------------
# System Prompt
# ---------------------------
SYSTEM_PROMPT = """
You are a helpful AI assistant who is specialized in resolving the user query.
You work on start, plan, action and observe mode.
For the given user query and the available tools, plan the step by step execution, and 
based on the planning select the relevant tools from the available tool.
And based on the tool selection you can perform the action to call the tool.
Wait for the observation and based on the observation from the tool call resolve the user query.

Rules:
- Follow the output in JSON format.
- Carefully analyze the user query
- Make decision and execute changes based on user's query.

Output JSON format:
{
    "step": "string",
    "content": "string",
    "function": "The name of the function if the step is the action",
    "input": "The input parameter of the function"
}

Available Tools:
- "get_weather": Takes {"city": "CityName"} and returns the current weather
- "run_command": Takes {"cmd": "your_command"} and executes the command
- "write_to_file": Takes {"filename": "file.txt", "content": "Hello"} and writes content to file
- "read_file": Takes {"filename": "file.txt"} and reads the file content
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# ---------------------------
# Main Loop
# ---------------------------
while True:
    text = input("Hi I am AI Assistant. How can I help you? ")
    print("🤖 Agent received query:", text)
    messages.append({"role": "user", "content": text})

    while True:
        response = client.chat.completions.create(
            model="gpt-5",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_response = json.loads(response.choices[0].message.content)
        messages.append({
            "role": "assistant",
            "content": json.dumps(parsed_response)
        })

        if parsed_response["step"] == "plan":
            print(f"🧠: {parsed_response['content']}")
            continue

        elif parsed_response["step"] == "action":
            tool_name = parsed_response["function"]
            tool_input = parsed_response["input"]

            # Ensure tool_input is always a dict
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {"value": tool_input}

            print(f"⛏️: Calling Tool: {tool_name} with the input {tool_input}")

            if tool_name in available_tools and available_tools[tool_name]:
                output = available_tools[tool_name](tool_input)
                print(output)
                messages.append({
                    "role": "user",
                    "content": json.dumps({"step": "observe", "output": output})
                })
                continue

        elif parsed_response["step"] == "output":
            print(f"✅ Final Answer: {parsed_response.get('output')}")
            break
