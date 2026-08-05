import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, dotenv_path: str = ".env", model = "gemini-3.5-flash-lite"):
        load_dotenv(dotenv_path)
        if "GOOGLE_API_KEY" not in os.environ:
            logger.error(f"No AI API key found")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,  
            max_retries=2
        )
        self.prompt = ChatPromptTemplate([
            ("system",
                "You are an AI query preprocessor for a CLIP-based visual search engine.\n"
                "Your task is to transform any user query into a concise, descriptive English text prompt optimized for vector similarity search.\n\n"
                "RULES:\n"
                "1. TRANSLATE: Always translate the query to English.\n"
                "2. HANDLE NEGATIONS: If the user excludes an object ('not X', 'without X'), completely omit X. "
                "DO NOT replace X with visually similar objects or close biological/physical relatives. "
                "Instead, describe broader positive categories, background environment, or distinct alternative concepts.\n"
                "3. VISUAL EXPANSION: Convert abstract feelings or vague phrases into concrete visual descriptors (lighting, setting, objects, mood).\n"
                "4. OUTPUT FORMAT: Return ONLY the final visual prompt string as comma-separated keywords or a concise sentence. No quotes, explanations, or chat filler."
            ),
            ("human", "{user_input}"),
                ])   
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("QueryProcessor chain successfully created.")

    def process_query(self, user_input: str) -> str:
        logger.info(f"Processing raw user query: '{user_input}'")
        cleaned_query = self.chain.invoke({"user_input": user_input})
        logger.info(f"Query processing complete. Transformed query: '{cleaned_query}'")
        return cleaned_query
        


if __name__ == "__main__":
    logger.info("=== Testing QueryProcessor Standalone ===")
    
    # Create processor instance
    processor = QueryProcessor()
    
    # Test cases with negations and complex structure
    test_queries = [
        "wild animals but not horses",
        "вечір на природі, не кімната",
        "show something cozy"
    ]
    
    for query in test_queries:
        print("\n" + "="*40)
        print(f"Original Query: {query}")
        result = processor.process_query(query)
        print(f"LLM Output:     {result}")