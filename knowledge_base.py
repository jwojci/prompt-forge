import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE = [
    # I. Finance
    {
        "domain": "Finance",
        "strategy": "Creative Writing",
        "prompt_id": "FIN-CW-01",
        "prompt_text": "You are a seasoned FinTech product innovator with a deep understanding of Millennial and Gen Z financial behaviors. Brainstorm a new financial product that gamifies savings and investing for this demographic. The product should be mobile-first and leverage social features. Provide a product name, a core value proposition, and three key features presented in a markdown table. The tone should be forward-thinking and exciting.",
        "explanation": "Encourages creative ideation by establishing a specific persona and a clear, innovative task. It guides the output format for easy parsing and sets a specific tone to influence the creative direction.",
    },
    {
        "domain": "Finance",
        "strategy": "Creative Writing",
        "prompt_id": "FIN-CW-02",
        "prompt_text": "Act as a marketing storyteller for a wealth management firm. Write a compelling, narrative-style case study about a fictional client named 'Alex', a 35-year-old entrepreneur who successfully grew their wealth using our firm's sustainable investing portfolio. The story should follow a classic narrative arc: Alex's initial financial challenges, the journey of working with our firm, and the ultimate positive outcome. The case study should be approximately 500 words and written in an engaging and persuasive tone.",
        "explanation": "Focuses on narrative creation for a specific marketing purpose. It provides a clear character, a desired story structure, and a target length, guiding the LLM to produce a well-formed and persuasive piece of content.",
    },
    {
        "domain": "Finance",
        "strategy": "Technical Explanation",
        "prompt_id": "FIN-TE-01",
        "prompt_text": "You are a quantitative analyst. Explain the process of using a Monte Carlo simulation for financial forecasting of a stock portfolio. Break down the explanation into the following sections: 1. Definition and Core Concept, 2. Key Input Variables (e.g., historical volatility, expected returns), 3. Step-by-Step Simulation Process, and 4. Interpretation of the Output Distribution. The explanation should be clear enough for an MBA student with a foundational understanding of finance. Use LaTeX for mathematical notations.",
        "explanation": "Requests a detailed technical explanation of a complex financial model. By specifying the target audience and a structured output, it ensures the explanation is both comprehensive and accessible. The request for LaTeX formatting ensures high-quality presentation of mathematical concepts.",
    },
    {
        "domain": "Finance",
        "strategy": "Technical Explanation",
        "prompt_id": "FIN-TE-02",
        "prompt_text": "You are a financial communications expert. I will provide you with a quarterly financial report for a publicly traded tech company. Your task is to summarize the key findings in plain, accessible language for a non-expert audience. The summary should be no more than 300 words and cover: 1. Overall revenue growth or decline, 2. Key drivers of performance, 3. Notable risks or challenges mentioned, and 4. The company's future outlook. Avoid jargon and use analogies where helpful.",
        "explanation": "Focuses on translating complex financial information into an easily understandable format. It sets clear constraints on length and content, and the instruction to avoid jargon is crucial for achieving the desired outcome.",
    },
    {
        "domain": "Finance",
        "strategy": "Concise Answer",
        "prompt_id": "FIN-CA-01",
        "prompt_text": "I need to calculate the Price-to-Earnings (P/E) ratio for a company. The current stock price is $150, and the earnings per share (EPS) for the last four quarters are $2.50, $2.75, $2.60, and $2.80. Provide only the calculated P/E ratio. Do not include any explanation.",
        "explanation": "Designed for a quick, factual response. The instruction to provide 'only the calculated P/E ratio' is a strong constraint that forces a concise answer, ideal for automated financial data dashboards.",
    },
    {
        "domain": "Finance",
        "strategy": "Chain-of-Thought",
        "prompt_id": "FIN-CoT-01",
        "prompt_text": "You are a senior risk analyst. A company is considering acquiring a smaller tech startup. I will provide details about both companies. Your task is to perform a step-by-step financial risk assessment. First, identify the key financial risks (e.g., valuation risk, integration risk, market risk). For each risk, explain your reasoning. Then, propose a mitigation strategy for each identified risk. Finally, provide an overall risk score on a scale of 1 to 10 (1 being low risk, 10 being high risk) and justify your score. Let's think step by step.",
        "explanation": "Explicitly invokes a Chain-of-Thought process by asking for a step-by-step analysis. It breaks down a complex assessment into logical components, forcing the LLM to reason through the problem before arriving at a conclusion.",
    },
    # II. Software Engineering
    {
        "domain": "Software Engineering",
        "strategy": "Creative Writing",
        "prompt_id": "SE-CW-01",
        "prompt_text": "Act as a creative technologist. Generate three novel software project ideas that address a societal need in the area of mental wellness. For each idea, provide a one-sentence pitch, a target user persona, and a key innovative feature that sets it apart from existing solutions. The ideas should be presented in a markdown list.",
        "explanation": "Stimulates creative thinking within a specific domain. By requesting multiple distinct ideas with a defined structure, it encourages divergent thinking and produces well-formed, easy-to-compare concepts.",
    },
    {
        "domain": "Software Engineering",
        "strategy": "Technical Explanation",
        "prompt_id": "SE-TE-01",
        "prompt_text": "You are a senior software engineer who writes excellent documentation. I will provide you with a Python function. Your task is to generate a comprehensive docstring for this function in the Google Python Style. The docstring should include: 1. A one-line summary. 2. A more detailed explanation of the function's purpose. 3. Descriptions for all arguments (Args:). 4. A description of the return value (Returns:). 5. An example of how to use the function (Example:).",
        "explanation": "Highly specific about the desired output format, which is crucial for consistency in a codebase. The persona assignment and the detailed breakdown of the required docstring components ensure a high-quality, structured explanation.",
    },
    {
        "domain": "Software Engineering",
        "strategy": "Technical Explanation",
        "prompt_id": "SE-TE-02",
        "prompt_text": "You are a support engineer with excellent communication skills. A user has encountered a '502 Bad Gateway' error. Explain this error to a non-technical user in simple, reassuring language. The explanation should be under 100 words and focus on what the user should do next. Avoid technical jargon.",
        "explanation": "Focuses on translating a technical concept for a lay audience. The constraints on length and the emphasis on actionable advice make the output practical and user-friendly.",
    },
    {
        "domain": "Software Engineering",
        "strategy": "Concise Answer",
        "prompt_id": "SE-CA-01",
        "prompt_text": "I have a detailed bug report. I need a concise summary for a project management tool. Extract the following information and present it as a JSON object: 'title' (a short, descriptive title of the bug), 'severity' (Low, Medium, High, Critical), and 'one_sentence_summary'.",
        "explanation": "Designed for data extraction and summarization into a machine-readable format. Requesting a JSON output makes it easy to integrate the LLM's response into other software systems.",
    },
    {
        "domain": "Software Engineering",
        "strategy": "Chain-of-Thought",
        "prompt_id": "SE-CoT-01",
        "prompt_text": "You are an expert debugger. I have a snippet of Python code that is not working as expected. I will provide the code, the expected output, and the actual output. Your task is to debug the code step by step. First, identify the most likely cause of the error. Then, explain your reasoning. Next, suggest a corrected version of the code. Finally, explain why your correction works. Let's think step by step to find the solution.",
        "explanation": "By explicitly asking for a step-by-step debugging process, this prompt encourages logical reasoning and a structured approach to problem-solving. This is far more effective than simply asking 'fix this code.'",
    },
    # III. Medicine
    {
        "domain": "Medicine",
        "strategy": "Creative Writing",
        "prompt_id": "MED-CW-01",
        "prompt_text": "You are a creative medical researcher with expertise in both immunology and neurology. Generate a novel, testable hypothesis that links a specific immunological pathway to the progression of a neurodegenerative disease like Alzheimer's. The hypothesis should be presented as a single, clear statement. Follow this with a brief (under 200 words) explanation of the potential mechanism and a suggested experimental approach to test the hypothesis.",
        "explanation": "Encourages creative and lateral thinking in a highly specialized field. By assigning a cross-disciplinary persona, it prompts the LLM to make novel connections between different areas of knowledge.",
    },
    {
        "domain": "Medicine",
        "strategy": "Creative Writing",
        "prompt_id": "MED-CW-02",
        "prompt_text": "You are a pediatric nurse with a talent for storytelling. Write a short, allegorical story for a 6-year-old child that explains how vaccines work. The story should feature friendly characters (e.g., 'Captain Antibody', 'the Sneaky Virus Invaders') and explain the concept of immune memory in a simple, non-frightening way. The story should be under 300 words.",
        "explanation": "Uses creative writing to explain a complex medical concept to a specific, non-technical audience (a child). This demonstrates the use of analogy and narrative for effective communication.",
    },
    {
        "domain": "Medicine",
        "strategy": "Technical Explanation",
        "prompt_id": "MED-TE-01",
        "prompt_text": "You are a certified medical coding and billing specialist. A patient has received a bill with the CPT code 99214. Explain what this code means in simple, clear language for the patient. Your explanation should cover the type of service this code represents and why it might have been used for their visit. Keep the explanation under 150 words.",
        "explanation": "Designed to demystify complex medical administrative information for a layperson. The persona and the specific task ensure an accurate and easy-to-understand explanation.",
    },
    {
        "domain": "Medicine",
        "strategy": "Concise Answer",
        "prompt_id": "MED-CA-01",
        "prompt_text": "A patient is taking Warfarin. Is it safe to prescribe them Ibuprofen for pain? Provide a 'Yes' or 'No' answer, followed by a one-sentence explanation of the potential interaction.",
        "explanation": "In a clinical setting, speed and accuracy are paramount. This prompt is designed for a quick, direct answer to a critical question, with a brief but essential explanation of the reasoning.",
    },
    {
        "domain": "Medicine",
        "strategy": "Chain-of-Thought",
        "prompt_id": "MED-CoT-01",
        "prompt_text": "You are a skilled diagnostician. A 45-year-old male presents with a two-week history of intermittent chest pain, shortness of breath, and fatigue. I will provide his vital signs, EKG results, and relevant medical history. Your task is to generate a differential diagnosis. First, list at least three possible diagnoses in order of likelihood. For each diagnosis, provide the supporting evidence from the patient's presentation. Then, suggest the next diagnostic steps for each possibility. Let's work through this case step by step.",
        "explanation": "Mimics the clinical reasoning process of a physician. By requiring a step-by-step breakdown of the differential diagnosis, it ensures a thorough and logical analysis of the provided medical data.",
    },
    # IV. Legal
    {
        "domain": "Legal",
        "strategy": "Creative Writing",
        "prompt_id": "LEG-CW-01",
        "prompt_text": "You are a seasoned litigator known for your persuasive flair. Draft a short, powerful closing argument for a fictional civil case. The case involves a small bakery suing a large corporation for trademark infringement. The argument should appeal to the jury's sense of fairness and community, not just legal technicalities. The tone should be passionate but professional. Keep it under 400 words.",
        "explanation": "This creative prompt focuses on persuasive writing within a specific professional context, guiding the LLM to generate content that balances logical reasoning with emotional appeal.",
    },
    {
        "domain": "Legal",
        "strategy": "Technical Explanation",
        "prompt_id": "LEG-TE-01",
        "prompt_text": "You are a law professor specializing in contract law. Explain the 'Parol Evidence Rule' to a first-year law student. Your explanation should include: 1. A clear definition of the rule. 2. The purpose of the rule. 3. The key exceptions to the rule, with a brief example for each. The explanation should be structured with clear headings for each section.",
        "explanation": "Requires a detailed and structured explanation of a complex legal concept. The specified audience and the required structure ensure the output is both comprehensive and easy to follow for a learner.",
    },
    {
        "domain": "Legal",
        "strategy": "Concise Answer",
        "prompt_id": "LEG-CA-01",
        "prompt_text": "Provide a concise, one-sentence definition of the legal term 'res judicata'.",
        "explanation": "Designed for a quick and accurate definition of a legal term of art. It's useful for situations where a brief, precise explanation is needed without extensive detail.",
    },
    {
        "domain": "Legal",
        "strategy": "Chain-of-Thought",
        "prompt_id": "LEG-CoT-01",
        "prompt_text": "You are a senior associate at a law firm. A client wants to know if they have a valid claim for breach of contract. I will provide the contract and a summary of the events that transpired. Your task is to analyze the situation step by step. First, identify the relevant clauses in the contract. Second, determine if the other party's actions constitute a breach of those clauses, explaining your reasoning. Third, assess any potential defenses the other party might raise. Finally, provide your overall assessment of the strength of the client's claim. Let's analyze this methodically.",
        "explanation": "Guides the LLM through a structured legal analysis. By breaking down the problem into distinct logical steps, it ensures a thorough and well-reasoned response that mirrors how a lawyer would approach the issue.",
    },
    # V. Marketing
    {
        "domain": "Marketing",
        "strategy": "Creative Writing",
        "prompt_id": "MKT-CW-01",
        "prompt_text": "You are a creative director at a cutting-edge ad agency. Develop a social media campaign concept for a new brand of sustainable, plant-based protein powder. The campaign should be targeted at environmentally conscious consumers aged 25-40. Provide a campaign hashtag, a core visual theme, and three example social media posts (for Instagram, TikTok, and X/Twitter) that embody the campaign's voice and message. The tone should be inspiring and authentic.",
        "explanation": "Designed to generate a complete, multi-platform marketing campaign concept. By specifying the target audience, tone, and deliverables, it guides the LLM to produce a cohesive and creative marketing plan.",
    },
    {
        "domain": "Marketing",
        "strategy": "Technical Explanation",
        "prompt_id": "MKT-TE-01",
        "prompt_text": "You are an SEO strategist. Explain the concept of 'keyword clustering' to a small business owner who is new to SEO. Your explanation should cover: 1. What keyword clustering is. 2. Why it is a more effective strategy than targeting single keywords. 3. A simple, step-by-step process for how to create keyword clusters. Use an analogy to make the concept easier to understand.",
        "explanation": "Focuses on making a technical marketing concept accessible to a non-expert. The requirement for an analogy is a key element that encourages a more intuitive and memorable explanation.",
    },
    {
        "domain": "Marketing",
        "strategy": "Concise Answer",
        "prompt_id": "MKT-CA-01",
        "prompt_text": "Generate 5 high-CTR headline variations for a blog post titled 'The Future of AI in Marketing'. Provide only a numbered list of the headlines. Do not include any introductory or concluding text.",
        "explanation": "This prompt is optimized for generating a list of creative options quickly and concisely. The constraint to provide 'only' the list makes the output clean and easy to use directly.",
    },
    {
        "domain": "Marketing",
        "strategy": "Chain-of-Thought",
        "prompt_id": "MKT-CoT-01",
        "prompt_text": "You are a senior marketing strategist. Your goal is to develop a digital marketing funnel for a new B2B SaaS product. Let's think step by step. First, define the three main stages of the funnel (e.g., Top-of-Funnel, Middle-of-Funnel, Bottom-of-Funnel). Second, for each stage, identify the primary goal (e.g., awareness, consideration, conversion). Third, for each stage, list at least two specific marketing channels and tactics you would use (e.g., LinkedIn ads, SEO-optimized blog posts, case study webinars, targeted email sequences). Present the final output as a markdown table.",
        "explanation": "Guides the LLM to build a complex strategy in a structured way. The step-by-step instruction ensures all parts of the funnel are considered logically and presented in a clear, organized format.",
    },
]


def create_vector_store(knowledge_base: list[dict], model: SentenceTransformer) -> None:
    """
    Generates embeddings for the knowledge base and creates a FAISS index.

    Args:
        knowledge_base_data (list): A list of dictionaries, where each dict represents a prompt.
        model: A pre-loaded SentenceTransformer model.
    """
    print("Extracting prompt texts from knowledge base...")
    prompt_texts = [
        item["prompt_text"] for item in knowledge_base
    ]  # Only prompt text needs to be vectorized

    print("Generating embeddings...")
    embeddings = model.encode(
        prompt_texts, convert_to_tensor=False, normalize_embeddings=True
    )

    # The dimension of our embeddings
    d = embeddings.shape[1]

    print(f"Embeddings generated with dimension: {d}")

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(d)

    # Add the vectorized prompts to the index
    index.add(embeddings)

    print(f"Index created successfully with {index.ntotal} entries.")

    # Save the index to disk
    faiss.write_index(index, "knowledge_base_index.bin")

    print("Index saved to disk.")


if __name__ == "__main__":
    print("Loading pre-trained sentence transformer model...")
    # Load the pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create the vector store
    create_vector_store(KNOWLEDGE_BASE, model)
