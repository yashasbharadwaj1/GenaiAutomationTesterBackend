import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from io import BytesIO
import cv2
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
import imageio
from moviepy import *

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

@app.post("/upload/pdf", response_model=UploadResponse, tags=["Generate Test Case"])
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


@app.post("/upload/video", response_model=UploadVideoDocResponse, tags=["Generate Test Case"])
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


@app.post("/prepare/testcase/and/store", response_model=PrepareTestCaseAndStoreResponse, tags=["Generate Test Case"])
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


@app.post("/retrieve/testcase/info", response_model=TestCaseExtractorResponse, tags=["Generate Test Case"])
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


@app.post("/execute/testcases", tags=["Execute Test Cases"])
async def execute_test_cases(test_cases_prompt: TestCasesPrompt):
    # Dump prompt data and run the agent.
    test_cases = test_cases_prompt.model_dump()
    agent_response = await run_agent(test_cases)

    item = agent_response.get("results")[0]
    project_name = item.get("project")
    test_suite_name = item.get("test_suite")

    temp_dir = os.path.abspath("tempdir")
    os.makedirs(temp_dir, exist_ok=True)

    # Convert GIF to Video mp4
    frames = imageio.mimread("agent_history.gif")
    clip = ImageSequenceClip(frames, fps=1)
    video_filename = f"{project_name}_{test_suite_name}.mp4"
    video_file_path = os.path.join(temp_dir, video_filename)
    clip.write_videofile(video_file_path, codec="libx264")

    # Upload the video file to Supabase.
    video_public_url = upload_file_to_supabase(video_file_path)

    # Remove the local video file after uploading.
    if os.path.exists(video_file_path):
        os.remove(video_file_path)

    agent_response["video_public_url"] = video_public_url

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
        "video_url": agent_response.get("video_public_url"),
    }

    supabase.table("results").insert(data).execute()
    return agent_response


@app.post("/upload/testing/steps", tags=["Existing Test Case Input"])
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
