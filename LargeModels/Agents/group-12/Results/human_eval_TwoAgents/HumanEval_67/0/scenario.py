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
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

assistant = autogen.AssistantAgent(
    "assistant",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    system_message="You are programming assistant. You should not print TERMINATE, unless you want to give up the task. And you don't need to explain your code, just output the code block as requested. After you give the code, I will run the code and give you feedback. If there are any errors, fix them and give me the new code.",
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
    },
    max_consecutive_auto_reply=5,
    default_auto_reply="TERMINATE",
)

user_proxy.initiate_chat(
    assistant,
    message="""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs
it on the function `fruit_distribution`. This will run a set of automated unit tests to verify the
correct implementation of `fruit_distribution`. However, `fruit_distribution` is only partially
implemented in the code below. Complete the implementation of `fruit_distribution` and output
a new stand-alone code block that contains everything needed to run the tests, including: importing
`my_tests`, calling `run_tests(fruit_distribution)`, as well as fruit_distribution's complete definition,
such that this code block can be run directly in Python.

```python
from my_tests import run_tests

"""
    + PROMPT
    + """

# Run the unit tests
run_tests(fruit_distribution)
```
""",
)

##############################
testbed_utils.finalize(agents=[assistant, user_proxy])
