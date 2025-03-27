import json
import logging
import os
import re
import tempfile
import time
import uuid
from io import BytesIO
import cv2
import git
import httpx
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from supabase import create_client, Client
from typing import Union
from base_models import *
from execute_test_case import run_agent
import requests
from werkzeug.utils import secure_filename

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


def process_response(response_text: str):
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")
    return data


ALLOWED_EXTENSIONS = {'.pdf', '.mp4'}


def upload_file_to_supabase(file: Union[str, UploadFile], bucket_name: str = "files") -> str:
    """
    Uploads a file (file path as str or UploadFile) to Supabase storage and returns the public URL.
    Only files with .pdf and .mp4 extensions are allowed.
    Ensures that file paths are within a secure base directory.
    """
    if isinstance(file, str):
        # Define a secure base directory.
        base_path = os.path.abspath("tempdir")
        file_path = os.path.abspath(file)
        # Validate that file_path is within the base_path.
        if not file_path.startswith(base_path + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path.")
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    elif hasattr(file, "filename") and hasattr(file, "file"):
        # This branch will work for FastAPI's UploadFile (or any similar object)
        filename = file.filename
        try:
            file_bytes = file.file.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type provided.")

    # Validate file extension.
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf and .mp4 files are allowed.")

    try:
        # Upload the file bytes to Supabase.
        supabase.storage.from_(bucket_name).upload(filename, file_bytes)
    except Exception as e:
        if "Duplicate" in str(e):
            raise HTTPException(status_code=400, detail="File already exists in storage.")
        else:
            raise HTTPException(status_code=500, detail=str(e))

    public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
    return public_url


@app.post("/upload/pdf", response_model=UploadResponse, tags=["Black Box Testing"])
def upload_pdf_file(
        file: UploadFile = File(...),
        project: str = Form(...),
        test_suite: str = Form(...)
):
    allowed_extension = ".pdf"
    if not file.filename.lower().endswith(allowed_extension):
        raise HTTPException(status_code=400, detail="Only PDF file is allowed.")

    logger.info("Received file: %s for project: %s and test_suite: %s", file.filename, project, test_suite)

    # Check if the file already exists for this project and test suite
    existing = supabase.table("documents") \
        .select("id, project, test_suite, blob_url") \
        .eq("project", project) \
        .eq("test_suite", test_suite) \
        .execute()

    if existing.data and len(existing.data) > 0:
        raise HTTPException(status_code=400, detail="File already exists in the knowledge base for this test suite.")

    blob_url = upload_file_to_supabase(file)

    # Store file metadata using both project and test_suite
    data = {
        "id": str(uuid.uuid4()),
        "project": project,
        "test_suite": test_suite,
        "blob_url": blob_url
    }
    supabase.table("documents").insert(data).execute()

    return {"status": "success", "filename": file.filename, "blob_url": blob_url}


# ---------------- Background Task: Process Video to PDF ----------------
def process_video_to_pdf(temp_file_path: str, project: str, test_suite: str):
    """
    Process a video by sampling frames, applying filters, and converting to a PDF.
    The resulting PDF is uploaded and its metadata stored with both project and test_suite.
    """
    pdf_filename = None  # ensure variable exists in finally block
    try:
        logger.info("Processing video for project: %s, test_suite: %s", project, test_suite)
        cap = cv2.VideoCapture(temp_file_path)
        frames = []
        frame_interval = 40  # sample every 40th frame
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                original_width = frame.shape[1]
                desired_width = 1280 if original_width >= 1280 else original_width
                aspect_ratio = frame.shape[0] / frame.shape[1]
                desired_height = int(desired_width * aspect_ratio)
                resized_frame = cv2.resize(frame, (desired_width, desired_height))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                sharpened_img = pil_img.filter(ImageFilter.SHARPEN)
                frames.append(sharpened_img.convert("RGB"))
            frame_count += 1

        cap.release()
        logger.info("Captured %d frames from video.", len(frames))
        if not frames:
            logger.error("No frames captured from video.")
            return

        # Use test_suite as the PDF filename
        pdf_filename = f"{test_suite}.pdf"
        frames[0].save(pdf_filename, "PDF", resolution=150.0, save_all=True, append_images=frames[1:])
        logger.info("PDF created: %s", pdf_filename)

        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
        dummy_file = UploadFile(filename=pdf_filename, file=BytesIO(pdf_bytes))
        pdf_url = upload_file_to_supabase(dummy_file)
        logger.info("PDF uploaded. URL: %s", pdf_url)

        # Insert record into the documents table with project and test_suite
        data = {
            "id": str(uuid.uuid4()),
            "project": project,
            "test_suite": test_suite,
            "blob_url": pdf_url
        }
        supabase.table("documents").insert(data).execute()

    except Exception as e:
        logger.error("Error processing video to PDF: %s", e)
    finally:
        time.sleep(0.5)
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as remove_err:
            logger.error("Error removing temp video file %s: %s", temp_file_path, remove_err)
        try:
            if pdf_filename and os.path.exists(pdf_filename):
                os.remove(pdf_filename)
        except Exception as remove_err:
            logger.error("Error removing PDF file %s: %s", pdf_filename, remove_err)


@app.post("/upload/video", response_model=UploadVideoDocResponse, tags=["Black Box Testing"])
def upload_video_file(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        project: str = Form(...),
        test_suite: str = Form(...)
):
    allowed_extension = ".mp4"
    if not file.filename.lower().endswith(allowed_extension):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed.")

    logger.info("Received video file: %s for project: %s and test_suite: %s", file.filename, project, test_suite)

    with tempfile.NamedTemporaryFile(delete=False, suffix=allowed_extension) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        temp_file_path = tmp.name

    background_tasks.add_task(process_video_to_pdf, temp_file_path, project, test_suite)
    return UploadVideoDocResponse(status="processing", project=project, test_suite=test_suite)


@app.post("/prepare/testcase/and/store", response_model=PrepareTestCaseAndStoreResponse, tags=["Black Box Testing"])
def prepare_test_case_and_store_it(request: PrepareTestCaseAndStoreRequest):
    # Retrieve document info using both project and test_suite

    result = supabase.table("documents") \
        .select("id, project, test_suite, blob_url") \
        .eq("project", request.project) \
        .eq("test_suite", request.test_suite) \
        .execute()

    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="Test suite not found in the knowledge base for this project.")

    blob_url = result.data[0]["blob_url"]

    # Download PDF content
    pdf_response = httpx.get(blob_url)
    if pdf_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to download the PDF from storage.")
    pdf_content = pdf_response.content

    prompt = (
        """You are provided with a PDF document that contains both textual content and visual elements (e.g., screenshots, diagrams, and other visual frames). Your task is to analyze the entire document and extract detailed test cases by integrating insights from both the text and the visuals. Please adhere to the following guidelines:

1. Identify and Process Visual Elements:
   - Detect every distinct visual frame or screenshot in the document.
   - For each visual element, determine if multiple scenarios (such as different UI movements or changes) are implied.
   

2. Extract and Organize Test Case Information: - Generate test cases from both textual content and visual cues. - 
**Scenario-Based Grouping:** For common scenarios (e.g., Login), if multiple related test cases are identified, 
combine them into a single, comprehensive scenario-based test case. In contrast, for test cases addressing distinct 
features or functionalities, create individual test cases. - Identify any visible error messages in the visuals. If 
errors are present, add them into your final result

3. Define Each Test Case with the Following Structure:
   - **Test Case Name:** A clear, descriptive title or identifier.
   - **Testing Steps:** A detailed, step-by-step list of manual testing actions that includes:
       - UI interactions or movements.
       - The application URL (if applicable).
   - **Expected Output:** A description of the expected results when the test case is executed.

4. Output Format:
   - Return the results as a valid JSON array.
   - Each element in the array must be an object with exactly the following keys: `test_case_name`, `testing_steps`, and `expected_output`.

Ensure that your final output is valid JSON and that it comprehensively covers all test cases derived from both the text and visual elements of the document, with appropriate grouping for common scenarios and separate test cases for individual features.
"""
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=pdf_content,
                mime_type='application/pdf',
            ),
            prompt
        ]
    )

    try:
        data = process_response(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AI response: {e}")

    # Delete old test cases for this project and test_suite
    supabase.table("testcases") \
        .delete() \
        .eq("project", request.project) \
        .eq("test_suite", request.test_suite) \
        .execute()

    # Insert new test cases with project and test_suite
    for test_case in data:
        row = {
            "project": request.project,
            "test_suite": request.test_suite,
            "test_case_name": test_case.get("test_case_name", ""),
            "testing_steps": test_case.get("testing_steps", ""),
            "expected_output": test_case.get("expected_output", "")
        }
        supabase.table("testcases").insert(row).execute()

    return {"status": "success", "data": data}


@app.post("/retrieve/testcase/info", response_model=TestCaseExtractorResponse, tags=["Black Box Testing"])
def retrieve_test_case_info(request: TestCaseExtractorRequest):
    result = supabase.table("testcases") \
        .select("project, test_suite, test_case_name, testing_steps, expected_output") \
        .eq("project", request.project) \
        .eq("test_suite", request.test_suite) \
        .execute()

    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="No test cases found for the provided project and test suite.")

    test_cases = []
    for row in result.data:
        steps = row.get("testing_steps", [])
        steps = json.loads(steps)
        test_cases.append(TestCase(
            project=row.get("project", ""),
            test_suite=row.get("test_suite", ""),
            test_case_name=row.get("test_case_name", ""),
            testing_steps=steps,
            expected_output=row.get("expected_output", "")
        ))
    return TestCaseExtractorResponse(test_cases=test_cases)


@app.post("/execute/testcases", tags=["Black Box Testing"])
async def execute_test_cases(test_cases_prompt: TestCasesPrompt):
    # Dump prompt data and run the agent.
    test_cases = test_cases_prompt.model_dump()
    agent_response = await run_agent(test_cases)

    item = agent_response.get("results")[0]
    project_name = item.get("project")
    test_suite_name = item.get("test_suite")

    # Delete any previous results for the same project and test suite.
    supabase.table("results") \
        .delete() \
        .eq("project", project_name) \
        .eq("test_suite", test_suite_name) \
        .execute()

    data = {
        "project": project_name,
        "test_suite": test_suite_name,
        "test_case_results": agent_response.get("results"),
    }

    supabase.table("results").insert(data).execute()
    return agent_response


@app.post("/upload/testing/steps", tags=["Black Box Testing"])
def upload_testing_steps(request: TestCase):
    row = {
        "project": request.project,
        "test_suite": request.test_suite,
        "test_case_name": request.test_case_name,
        "testing_steps": request.testing_steps,
        "expected_output": request.expected_output
    }
    try:
        # Check if a test case already exists for the given project, test_suite, and test_case_name.
        existing = supabase.table("testcases") \
            .select("id") \
            .eq("project", request.project) \
            .eq("test_suite", request.test_suite) \
            .eq("test_case_name", request.test_case_name) \
            .execute()

        if existing.data and len(existing.data) > 0:
            # If record exists, update it with the new details.
            supabase.table("testcases") \
                .update(row) \
                .eq("project", request.project) \
                .eq("test_suite", request.test_suite) \
                .eq("test_case_name", request.test_case_name) \
                .execute()
        else:
            # If no record exists, insert the new test case.
            supabase.table("testcases").insert(row).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "data": row}


# Define a safe root for all repository paths.
SAFE_ROOT = os.path.abspath("repositories")
if not os.path.exists(SAFE_ROOT):
    os.makedirs(SAFE_ROOT)

def get_safe_repo_path(repo_name: str) -> str:
    """
    Normalize and validate the repository name to ensure the path is safe.
    The repository will be stored under the SAFE_ROOT directory.
    """
    sanitized_repo_name = secure_filename(repo_name)
    normalized = os.path.normpath(sanitized_repo_name)
    full_path = os.path.join(SAFE_ROOT, normalized)
    # Ensure that the full path is within the safe root.
    if not os.path.abspath(full_path).startswith(SAFE_ROOT):
        raise HTTPException(status_code=400, detail="Invalid repository name.")
    return full_path

def parse_github_repo(repo_url: str):
    """
    Parses a GitHub repo URL to extract the owner and repository name.
    """
    # Remove trailing slash and .git suffix if present
    repo_url = repo_url.rstrip("/").replace(".git", "")
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, repo_url)
    if match:
        owner = match.group(1)
        repo_name = match.group(2)
        return owner, repo_name
    return None, None

def check_repo_privacy(owner: str, repo_name: str, username: str = None, pat_token: str = None):
    """
    Uses GitHub API to determine if a repository is private.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    auth = None
    if pat_token:
        # For private repo API access, credentials are needed.
        auth = (username if username else "", pat_token)
    response = requests.get(api_url, auth=auth)
    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get("private", False)
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error accessing repository info: {response.text}"
        )

def clone_repo(repo_url: str, destination_folder: str, branch: str = None):
    """
    Clones the repository into destination_folder if it does not exist.
    If a branch is specified, clones that branch; otherwise, clones the default branch.
    """
    if not os.path.exists(destination_folder):
        try:
            if branch:
                git.Repo.clone_from(repo_url, destination_folder, branch=branch)
            else:
                git.Repo.clone_from(repo_url, destination_folder)
            return f"Repository cloned successfully into '{destination_folder}'."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error cloning repository: {e}")
    else:
        return f"Directory '{destination_folder}' already exists. Skipping clone."

@app.post("/clone/repo", tags=["White Box Testing"])
def clone_github_repo(clone_request: CloneRequest):
    repo_url = clone_request.repo_url
    branch = clone_request.branch

    # Parse the repo URL to extract owner and repository name.
    owner, repo_name = parse_github_repo(repo_url)
    if not owner or not repo_name:
        raise HTTPException(status_code=400, detail="Invalid GitHub repository URL format.")

    # Get a safe destination folder within SAFE_ROOT.
    destination_folder = get_safe_repo_path(repo_name)

    # Determine if the repository is private by calling the GitHub API.
    try:
        is_private = check_repo_privacy(owner, repo_name, clone_request.username, clone_request.pat_token)
    except HTTPException as e:
        raise e

    # For private repos, ensure that a PAT token is provided and modify the repo URL accordingly.
    if is_private:
        if not clone_request.pat_token:
            raise HTTPException(
                status_code=401,
                detail="Private repository requires a Personal Access Token (PAT)."
            )
        if repo_url.startswith("https://"):
            user = clone_request.username if clone_request.username else ""
            repo_url = repo_url.replace("https://", f"https://{user}:{clone_request.pat_token}@")

    message = clone_repo(repo_url, destination_folder, branch)
    return {"message": message, "destination": destination_folder}

def get_code_from_files(folder, extensions=None):
    """
    Recursively reads code files from the given folder.
    """
    if extensions is None:
        extensions = ['.py', '.js', '.java', '.cpp', '.c', '.ts']
    code_snippets = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        code_snippets[file_path] = code
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
    return code_snippets

def aggregate_code(code_snippets):
    """
    Aggregates code from all files, adding file context.
    """
    aggregated = ""
    for file_path, code in code_snippets.items():
        aggregated += f"File: {file_path}\n{code}\n\n"
    return aggregated

@app.post("/analyze/code_review", tags=["White Box Testing"])
def analyze_code_review(request: CodeReviewRequest):
    # Validate and get the safe repository folder.
    folder = get_safe_repo_path(request.repo_name)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Repository folder '{folder}' not found.")

    # Read and aggregate code from the repository folder.
    code_files = get_code_from_files(folder)
    aggregated_code = aggregate_code(code_files)

    # System instruction for factor-based code review analysis.
    system_instruction = r"""
You are a code review agent. You will be given a large chunk of code.
Please analyze the code for the following factors and return the result in valid JSON format:
{
  "Code Coverage": "<analysis>",
  "Logic Flaws": "<analysis>",
  "Complexity": "<analysis>",
  "Input Validation": "<analysis>",
  "Security Issues": "<analysis>",
  "Error Handling": "<analysis>",
  "Code Style & Best Practices": "<analysis>",
  "Performance Hotspots": "<analysis>"
}
Analyze the code based on:
- Code Coverage: Line, branch, function, and path coverage.
- Logic Flaws: Incorrect logic, dead code, unreachable conditions.
- Complexity: Cyclomatic complexity, deeply nested conditions.
- Input Validation: Lack of input sanitization, boundary conditions.
- Security Issues: SQL injection, XSS, hardcoded secrets, etc.
- Error Handling: Missing try/catch, poor exception management.
- Code Style & Best Practices: Lint errors, naming, conventions.
- Performance Hotspots: Inefficient loops, unnecessary database calls.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=aggregated_code
        )
        code_review_result = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {e}")

    # Assume process_response is a function that cleans or parses the Gemini output.
    processed_code_review_result = process_response(code_review_result)
    return {"repo":folder,"result": processed_code_review_result}

@app.post("/generate/unit_tests", tags=["White Box Testing"])
def generate_unit_tests(request: UnitTestRequest):
    # Validate and get the safe repository folder.
    folder = get_safe_repo_path(request.repo_name)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Repository folder '{folder}' not found.")

    # Read and aggregate code from the repository folder.
    code_files = get_code_from_files(folder)
    aggregated_code = aggregate_code(code_files)

    system_instruction = r"""
You are a unit test generator for a Python codebase. 
Your task is to analyze the provided repository code and generate comprehensive, copy-paste ready unit tests for each file.
Ensure that the tests follow best practices for Python unit testing.
Return the complete unit test code as a single Python string enclosed within triple double quotes.
Do not include any additional text, explanations, or JSON formatting—only the unit test code.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            contents=aggregated_code
        )
        unit_testing_result = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {e}")

    logger.info("unit testing result: %s", unit_testing_result)
    return {"repo": folder, "result": unit_testing_result}
