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
config_list = [{"model":"qw2.5","base_url":"http://172.17.0.1:8000/v1","api_key":"EMPTY"}]

planner = autogen.AssistantAgent(
    "planner",
    system_message="""
    You are a seasoned software architect and algorithm designer. Your task is to conduct an in-depth analysis of the HumanEval programming problem and formulate a detailed implementation plan.Your responsibilities:Analyze problem requirements and constraints Design algorithm ideas and data structures Formulate implementation strategies and steps Identify potential challenges and edge cases

    Propose test case design ideas

    Please output the plan in the following format:

    Requirements Analysis
    Key requirements

    Input/output specifications

    Algorithm Design
    Core algorithm approach

    Time complexity analysis

    Implementation Strategy
    Step-by-step breakdown

    Key function design
    """,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

engineer = autogen.AssistantAgent(
    "engineer",
    system_message="""
    You are an experienced software engineer. Your task is to write high-quality, executable Python code based on requirements analysis, algorithm design, implementation strategies, and test plans.

    Your responsibilities:

    Implement code strictly according to the plan.

    Write Python code that complies with PEP8 standards.

    Include necessary comments and docstrings.

    Implement complete function logic.

    Incorporate basic error handling.

    Code requirements:

    Function signatures must exactly match the problem requirements.

    Include complete function implementations.

    Code should be concise and efficient.

    Include appropriate comments.
    """,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

debugger = autogen.AssistantAgent(
    "debugger",
    system_message="""
    You are a rigorous code review expert. Your task is to conduct a comprehensive review and testing analysis of the implemented code.

    Your responsibilities:

    Code quality review (style, readability, efficiency)

    Functional correctness analysis

    Boundary case handling check

    Potential bug identification

    Please output the review results in the following format:

    Code Quality Score (1-10 points)
    Score: X

    Strengths: ...

    Weaknesses: ...

    Functional Correctness
    Meets requirements: Yes/No

    Potential issues: ...

    Improvement Suggestions
    Specific suggestion 1

    Specific suggestion 2
    """,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

fixer = autogen.AssistantAgent(
    "fixer",
    system_message="""
    You are a professional code debugging and optimization expert. Your task is to fix issues and optimize the code based on review feedback (including code quality score (out of 10), functional correctness, improvement suggestions, and testing recommendations).

    Your responsibilities:

    Analyze review comments and understand the issues

    Fix bugs and defects in the code

    Optimize code performance and readability

    Ensure the fixed code fully meets requirements

    Fix requirements:

    Only modify problematic parts while preserving good code

    Ensure the fixed code passes all tests
    """,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

checker = autogen.AssistantAgent(
    "checker",
    system_message="""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs
it on the function `triangle_area`. This will run a set of automated unit tests to verify the
correct implementation of `triangle_area`. However, `triangle_area` is only partially
implemented in the code below. Based on a comprehensive consideration of all the above information, complete the implementation of `triangle_area` and output
a new stand-alone code block that contains everything needed to run the tests, including: importing
`my_tests`, calling `run_tests(triangle_area)`, as well as triangle_area's complete definition,
such that this code block can be run directly in Python.

```python
from my_tests import run_tests

"""
    + PROMPT
    + """

# Run the unit tests
run_tests(triangle_area)
```
""",
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    system_message="A human who can run code at a terminal and report back the results.",
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
        "last_n_messages": "auto",
        "timeout": 20,
    },
)

groupchat = autogen.GroupChat(
    agents=[planner, engineer, debugger, fixer, checker, user_proxy],
    messages=[],
    speaker_selection_method="round_robin",
    max_round=7,
)


manager = autogen.GroupChatManager(
    groupchat=groupchat
)

user_proxy.initiate_chat(
    manager,
    message="""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs
it on the function `triangle_area`. This will run a set of automated unit tests to verify the
correct implementation of `triangle_area`. However, `triangle_area` is only partially
implemented in the code below. Complete the implementation of `triangle_area` and output
a new stand-alone code block that contains everything needed to run the tests, including: importing
`my_tests`, calling `run_tests(triangle_area)`, as well as triangle_area's complete definition,
such that this code block can be run directly in Python.

```python
from my_tests import run_tests

"""
    + PROMPT
    + """

# Run the unit tests
run_tests(triangle_area)
```
""",
)


##############################
testbed_utils.finalize(agents=[planner, engineer, debugger, fixer, checker, user_proxy, manager])
