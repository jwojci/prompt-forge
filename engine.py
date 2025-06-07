import os
import json
import logging

import faiss
from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types

import streamlit as st
from pydantic import BaseModel

from knowledge_base import KNOWLEDGE_BASE  # Import KNOWLEDGE_BASE at the top

# LOGGER CONFIG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="engine.log",
)
logger = logging.getLogger(__name__)

# GEMINI CONFIG
try:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client initialized successfully.")

    # RAG Components
    logger.info("Loading RAG components...")
    RAG_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    FAISS_INDEX = faiss.read_index("knowledge_base_index.bin")
    logger.info(
        f"RAG components loaded successfully. Index has {FAISS_INDEX.ntotal} entries."
    )

except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {e}")
    GEMINI_CLIENT = None
    RAG_MODEL = None
    FAISS_INDEX = None


SYSTEM_PROMPTS = {
    "Creative Writing": """You are PromptForge, an AI assistant that rewrites user prompts. Your persona for this task is an expert creative writer with a talent for storytelling. Your goal is to transform a user's basic idea into a rich, detailed, and effective prompt for another AI.

    Focus on embedding these principles into the rewritten prompt:
    - Establishing a clear persona or voice for the final AI
    - Using narrative techniques and storytelling elements
    - Creating engaging and memorable content
    - Maintaining a consistent tone throughout
    - Structuring the output in a clear, organized way

    Your final output must be ONLY the rewritten prompt. DO NOT WRITE ANYTHING ELSE.""",
    "Technical Explanation": """You are PromptForge, an AI assistant that rewrites user prompts. Your persona for this task is a technical expert with exceptional communication skills. Your goal is to transform a user's technical query into a clear, structured, and comprehensive prompt for another AI.

    Focus on embedding these principles into the rewritten prompt:
    - Breaking down complex information into digestible parts
    - Using clear, precise language 
    - Providing structured explanations with clear sections
    - Including relevant examples or analogies
    - Adapting the technical depth to the target audience

    Your final output must be ONLY the rewritten prompt. DO NOT WRITE ANYTHING ELSE.""",
    "Concise Answer": """You are PromptForge, an AI assistant that rewrites user prompts. Your persona for this task is a precision-focused communicator. Your goal is to transform a user's request into a highly constrained prompt designed to elicit a brief, direct response from another AI.

    Focus on embedding these principles into the rewritten prompt:
    - Delivering only the essential information
    - Using clear, unambiguous language
    - Maintaining accuracy while being brief
    - Adding strong constraints to prevent verbosity (e.g., 'Provide only the answer')
    - Following any specific format requirements

    Your final output must be ONLY the rewritten prompt. DO NOT WRITE ANYTHING ELSE.""",
    "Chain-of-Thought": """You are PromptForge, an AI assistant that rewrites user prompts. Your persona for this task is a methodical problem-solver. Your goal is to transform a user's complex problem into a step-by-step prompt that forces another AI to show its reasoning process.

    Focus on embedding these principles into the rewritten prompt:
    - Explicitly instructing the AI to 'think step by step' or 'work through this methodically'
    - Breaking down the problem into logical sub-questions
    - Asking for reasoning at each step
    - Building toward a well-reasoned conclusion
    - Making the thought process transparent

    Your final output must be ONLY the rewritten prompt. DO NOT WRITE ANYTHING ELSE.""",
}


def retrieve_relevant_examples(query, k=3):
    """
    Retrieves the most relevant examples from the knowledge base based on the query.

    Args:
        query (str): The query to search for relevant examples.
        k (int): The number of relevant examples to retrieve.

    Returns:
        list: A list of relevant examples.
    """
    if not RAG_MODEL or not FAISS_INDEX:
        logging.error("ERROR: RAG/Vector Search not initialized.")
        return []

    try:
        logging.info(f"Retrieving {k} relevant examples for query: {query}")

        # Encode the query
        query_embedding = RAG_MODEL.encode([query], normalize_embeddings=True)

        # Search for the most similar examples
        distances, indices = FAISS_INDEX.search(query_embedding.astype("float32"), k)

        # Retrieve the relevant examples and format them
        relevant_examples = []
        for i in indices[0]:
            # Ensure the index is valid
            if 0 <= i < len(KNOWLEDGE_BASE):
                example = KNOWLEDGE_BASE[i]
                formatted_example = (
                    f"Example (from {example['domain']}/{example['strategy']}):\n"
                    f"Prompt: {example['prompt_text']}\n"
                    f"Explanation: {example['explanation']}\n"
                )
                relevant_examples.append(formatted_example)

        return relevant_examples
    except Exception as e:
        logging.error(f"Exception occurred in retrieve_relevant_examples: {e}")
        return []


def refine_prompt(user_prompt: str, system_prompt: str, retrieved_examples: str) -> str:
    """
    Refines a user prompt based on the specified strategy and retrieved examples.

    Args:
        user_prompt (str): The original user prompt to be refined.
        strategy (str): The strategy to use for refining the prompt.
        retrieved_examples (str): Examples of prompts that have been refined using the same strategy.

    Returns:
        str: The refined prompt.
    """
    if not GEMINI_CLIENT:
        logging.error("ERROR: GEMINI CLIENT not initialized.")
        return ""

    try:
        prompt_for_refiner = f"""
        **User's Basic Prompt:**
        "{user_prompt}"

        **Examples of High-Quality Prompts to Learn From:**
        {retrieved_examples}
        """

        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=prompt_for_refiner,
        )

        return response.text.strip()

    except Exception as e:
        logger.error(f"Exception occurred in refine_prompt: {e}")
        return ""


class JudgeOutput(BaseModel):
    score_A: int
    score_B: int


def evaluate_outputs(
    original_output: str, refined_output: str, user_prompt: str
) -> dict:
    """
    Evaluates and scores the original and refined outputs using a "Judge LLM".

    Args:
        original_output (str): The output generated from the original prompt.
        refined_output (str): The output generated from the refined prompt.
        user_prompt (str): The original user prompt that started the process.

    Returns:
        dict: A dictionary containing 'score_A' and 'score_B', or default scores on error.
    """
    if not GEMINI_CLIENT:
        logging.error("Cannot evaluate outputs, Gemini client is not available.")
        return {"score_A": 0, "score_B": 0}

    judge_prompt_template = f"""
    You are an impartial and meticulous AI Quality Analyst. Your task is to evaluate two AI-generated responses based on a user's request. You must compare them on the following criteria:
    1.  **Relevance**: How well does the response address the user's core request?
    2.  **Clarity**: Is the response clear, well-structured, and easy to understand?
    3.  **Detail & Completeness**: Does the response provide an appropriate level of detail to be useful?

    You will score each response on a scale of 1 to 10, where 1 is poor and 10 is excellent. Your response MUST be a valid JSON object and nothing else.

    ---
    **User's Request:**
    "{user_prompt}"

    ---
    **Response A (from original prompt):**
    "{original_output}"

    ---
    **Response B (from refined prompt):**
    "{refined_output}"
    ---

    Now, provide your evaluation. Respond ONLY with a JSON object containing two keys: "score_A" and "score_B".
    """

    try:
        logging.info("Sending request to Gemini API for evaluation.")

        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
                "response_schema": JudgeOutput,
            },
            contents=judge_prompt_template,
        )

        # The response text should be a JSON string.
        response_json = json.loads(response.text)

        # Basic validation to ensure the keys exist and are integers
        score_a = int(response_json.get("score_A", 0))
        score_b = int(response_json.get("score_B", 0))

        logging.info(f"Evaluation successful: Score A={score_a}, Score B={score_b}")
        return {"score_A": score_a, "score_B": score_b}

    except json.JSONDecodeError as e:
        logging.error(
            f"Failed to parse JSON from evaluation response: {e}\nResponse text: {response.text}"
        )
        return {"score_A": 0, "score_B": 0}
    except Exception as e:
        logging.error(f"An exception occurred during evaluation: {e}")
        return {"score_A": 0, "score_B": 0}


@st.cache_data
def get_llm_response(prompt: str) -> str:
    """
    Generates a response from an LLM based on the given prompt.

    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        str: The LLM's generated response.
    """
    if GEMINI_API_KEY is None:
        logger.error("ERROR: Gemini API key not available for LLM response generation.")
        return "Error: LLM API not configured."

    try:
        #     response = GEMINI_CLIENT.models.generate_content(
        #     model="gemini-2.0-flash",
        #     config=types.GenerateContentConfig(system_instruction=system_prompt),
        #     contents=prompt_for_refiner,
        # )
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(
            f"Exception occurred in get_llm_response with prompt '{prompt[:50]}...': {e}"
        )
        return f"Error generating response: {e}"
