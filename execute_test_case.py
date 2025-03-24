import json
import re
import asyncio
from datetime import datetime
import os
import logging

import truststore

truststore.inject_into_ssl()

from browser_use import Agent, Browser, Controller, BrowserConfig
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
controller = Controller()

config = BrowserConfig(
    headless=True,
    disable_security=True,
    extra_chromium_args=[
	'--headless=new',
	'--no-sandbox',
	'--disable-dev-shm-usage'	
    ],
    proxy=None
)
browser = Browser(config=config)


async def run_agent(test_cases_data):
    test_cases = test_cases_data.get("test_cases", [])
    instruction_line = (
        "You must execute every test case sequentially without skipping any. "
        "Each dictionary in the 'test_cases' array represents a separate test case that must be fully executed. "
        "After completing all test cases, output your final result in the exact format below, one test case per line, "
        "with only the test case number and its status (either 'Passed' or 'Failed'). For example:\n"
        "Test Case 1: Passed\n"
        "Test Case 2: Failed\n"
        "Test Case 3: Passed\n"
        "Do not include any additional commentary or details."
    )

    test_cases_data["user_instruction"] = instruction_line
    task = json.dumps(test_cases_data, indent=2)

    a = Agent(task=task, llm=llm, use_vision=True, controller=controller)
    history = await a.run()
    final_result = history.final_result()
    logger.info("Final Result is: %s", final_result)

    # Process test cases from the final result.
    final_test_cases = []
    for test_case in test_cases:
        test_case_dict = {
            "project": test_case.get("project"),
            "test_suite": test_case.get("test_suite"),
            "test_case_name": test_case.get("test_case_name")
        }
        final_test_cases.append(test_case_dict)

    result_lines = final_result.splitlines()
    for idx, test_case_dict in enumerate(final_test_cases):
        if idx < len(result_lines):
            line = result_lines[idx]
            # Expecting format: "Test Case X: Passed" or "Test Case X: Failed"
            match = re.match(r"Test Case\s+\d+:\s*(Passed|Failed)", line)
            if match:
                test_case_dict["test_result"] = match.group(1)
            else:
                test_case_dict["test_result"] = "Unknown"
        else:
            test_case_dict["test_result"] = "Not Executed"

    response = {
        "results": final_test_cases
    }

    logger.info("Final JSON Result: %s", response)
    return response


