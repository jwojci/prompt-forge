import streamlit as st
import json
import os
import logging
from dotenv import load_dotenv

from engine import (
    retrieve_relevant_examples,
    refine_prompt,
    evaluate_outputs,
    get_llm_response,
    SYSTEM_PROMPTS,
)

# Load environment variables
load_dotenv()

# Configure logging for app.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# --- UI Layout ---

st.set_page_config(
    page_title="PromptForge v1.0",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✨ PromptForge v1.0")
st.markdown(
    """
    **Forge state-of-the-art prompts with RAG-powered refinement and automated evaluation.**
    Enter your basic prompt, select a refinement strategy, and let PromptForge demonstrate its efficacy.
    """
)

# Sidebar for History
with st.sidebar:
    st.header("Session History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Prompt {len(st.session_state.history) - i}: {entry['user_prompt'][:30]}..."):
                st.write(f"**Strategy:** {entry['strategy']}")
                st.write(f"**Original Prompt:** `{entry['user_prompt']}`")
                st.write(f"**Refined Prompt:** `{entry['refined_prompt']}`")
                st.write(f"**Original Output Score:** {entry['original_score']}")
                st.write(f"**Refined Output Score:** {entry['refined_score']}")
                st.metric(
                    "Quality Uplift",
                    value=f"{entry['refined_score']}",
                    delta=f"{entry['refined_score'] - entry['original_score']}",
                    delta_color="normal",
                )
    else:
        st.info("No history yet. Forge some prompts!")

# Main content area
user_prompt = st.text_area(
    "Enter your prompt here:",
    placeholder="e.g., Write a story about a brave knight.",
    height=150,
)

strategies = list(SYSTEM_PROMPTS.keys())
selected_strategy = st.selectbox(
    "Select a refinement strategy:",
    options=strategies,
    index=0,  # Default to the first strategy
)

forge_button = st.button("Forge Prompt & Evaluate")

if forge_button and user_prompt:
    st.info("Forging your prompt, generating responses, and evaluating... This may take a moment.")
    logger.info(f"User initiated forging for prompt: {user_prompt[:50]}...")

    with st.spinner("Step 1/5: Retrieving relevant examples (RAG)..."):
        # 1. Retrieve relevant examples using RAG
        retrieved_examples_list = retrieve_relevant_examples(user_prompt, k=3)
        retrieved_examples_str = "\n\n".join(retrieved_examples_list) if retrieved_examples_list else "No relevant examples found in knowledge base."
        logger.info(f"Retrieved {len(retrieved_examples_list)} examples.")

    with st.spinner("Step 2/5: Refining the prompt..."):
        # 2. Refine the prompt
        system_instruction_for_refiner = SYSTEM_PROMPTS.get(selected_strategy)
        if not system_instruction_for_refiner:
            st.error("Invalid strategy selected. Please choose from the available options.")
            logger.error(f"Invalid strategy selected: {selected_strategy}")
            st.stop()
            
        refined_prompt = refine_prompt(
            user_prompt=user_prompt,
            system_prompt=system_instruction_for_refiner,
            retrieved_examples=retrieved_examples_str,
        )
        if not refined_prompt:
            st.warning("Prompt refinement failed. Please check backend logs.")
            logger.warning("Refined prompt is empty.")
            refined_prompt = user_prompt # Fallback to original prompt
        logger.info("Prompt refined successfully.")

    with st.spinner("Step 3/5: Generating response with original prompt..."):
        # 3. Generate response with original prompt
        original_output = get_llm_response(user_prompt)
        if not original_output:
            st.warning("Failed to generate output with original prompt.")
            logger.warning("Original output is empty.")
        logger.info("Original output generated.")


    with st.spinner("Step 4/5: Generating response with refined prompt..."):
        # 4. Generate response with refined prompt
        refined_output = get_llm_response(refined_prompt)
        if not refined_output:
            st.warning("Failed to generate output with refined prompt.")
            logger.warning("Refined output is empty.")
        logger.info("Refined output generated.")

    with st.spinner("Step 5/5: Evaluating outputs..."):
        # 5. Automated Evaluation
        scores = evaluate_outputs(original_output, refined_output, user_prompt)
        original_score = scores.get("score_A", 0)
        refined_score = scores.get("score_B", 0)
        logger.info(f"Evaluation complete: Original Score={original_score}, Refined Score={refined_score}")

    st.success("Process complete! See results below.")

    # Display Results
    st.subheader("Comparison & Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Prompt Output")
        st.info(original_output)
        st.metric("Original Score", value=original_score)

    with col2:
        st.markdown("### Refined Prompt Output")
        st.success(refined_output)
        st.metric(
            "Refined Score",
            value=refined_score,
            delta=refined_score - original_score,
            delta_color="normal",
        )

    # Use expander for detailed info
    with st.expander("Show Refined Prompt and RAG Details"):
        st.markdown(f"**Refinement Strategy:** `{selected_strategy}`")
        st.markdown(f"**Refined Prompt:**")
        st.code(refined_prompt, language='markdown')
        st.markdown(f"**Retrieved Examples (for RAG):**")
        if retrieved_examples_list:
            for i, example in enumerate(retrieved_examples_list):
                st.text_area(f"Example {i+1}", example, height=200, key=f"example_{i}")
        else:
            st.info("No specific examples were used from the knowledge base for refinement.")


    # Update session history
    st.session_state.history.append(
        {
            "user_prompt": user_prompt,
            "strategy": selected_strategy,
            "refined_prompt": refined_prompt,
            "original_output": original_output,
            "refined_output": refined_output,
            "original_score": original_score,
            "refined_score": refined_score,
        }
    )
    logger.info("Session history updated.")

elif forge_button and not user_prompt:
    st.warning("Please enter a prompt to begin.")
