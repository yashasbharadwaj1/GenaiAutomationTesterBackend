from typing import List, Optional, Any, Dict

from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    filename: str
    blob_url: str

class UploadVideoDocResponse(BaseModel):
    status: str
    project: str
    test_suite: str



class PrepareTestCaseAndStoreRequest(BaseModel):
    project: str
    test_suite: str


class PrepareTestCaseAndStoreResponse(BaseModel):
    data: List[Dict[str, Any]]
    status: str


class TestCase(BaseModel):
    project: str
    test_suite: str
    test_case_name: str
    testing_steps: List[str]
    expected_output: str

class TestCaseExtractorRequest(BaseModel):
    project: str
    test_suite: str

class TestCaseExtractorResponse(BaseModel):
    test_cases: List[TestCase]

class TestCasesPrompt(BaseModel):
    test_cases: List[dict]


