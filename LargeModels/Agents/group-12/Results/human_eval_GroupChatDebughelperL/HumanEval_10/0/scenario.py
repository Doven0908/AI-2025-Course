import base64
import json
import os

import testbed_utils

import autogen

# NOTE:
# This scenario runs Human Eval in a slightly unconventional way:
# The agents have access to the unit tests, and can keep trying
# until they pass.

testbed_utils.init()
##############################

work_dir = "coding"

# Read the prompt
PROMPT = ""
with open("prompt.txt", "rt") as fh:
    PROMPT = fh.read()

# Ok, now get autogen to solve it.
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST","",{"model":["qwen2.5-coder:7b"]})
config_list2 = autogen.config_list_from_json("OAI_CONFIG_LIST","",{"model":["qwen2.5-coder:3b"]})

assistant = autogen.AssistantAgent(
    "coder",
    system_message="You are programming assistant. You should not print TERMINATE, unless you want to give up the task. And you don't need to explain your code, just output the code block as requested. After you give the code, you will get feedbacks. You need fix bugs and give you code.",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    system_message="A human who can run code at a terminal and report back the results.",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
        "last_n_messages": "auto",
        "timeout": 20,
    },
    max_consecutive_auto_reply=5,
)

debugger_agent = autogen.AssistantAgent(
    "degub_agent",
    system_message="You are debugging assistant. You should not print TERMINATE, unless you want to give up the task. You need to analyse why the code is failing based on the error message, and give your advice to the coder agent to fix the problem.",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list2, timeout=180),
)

groupchat = autogen.GroupChat(
    agents=[assistant, user_proxy, debugger_agent],
    messages=[],
    speaker_selection_method="round_robin",
)


manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

debugger_agent.initiate_chat(
    manager,
    message="""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs
it on the function `make_palindrome`. This will run a set of automated unit tests to verify the
correct implementation of `make_palindrome`. However, `make_palindrome` is only partially
implemented in the code below. Complete the implementation of `make_palindrome` and output
a new stand-alone code block that contains everything needed to run the tests, including: importing
`my_tests`, calling `run_tests(make_palindrome)`, as well as make_palindrome's complete definition,
such that this code block can be run directly in Python.

```python
from my_tests import run_tests

"""
    + PROMPT
    + """

# Run the unit tests
run_tests(make_palindrome)
```
""",
)


##############################
testbed_utils.finalize(agents=[assistant, user_proxy, debugger_agent, manager])
