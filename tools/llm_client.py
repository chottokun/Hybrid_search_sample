from deep_research_project.config.config import Configuration
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, config: Configuration):
        self.config = config
        self.llm = None # Placeholder for the actual LLM client

        if self.config.LLM_PROVIDER == "openai":
            try:
                from langchain_openai import ChatOpenAI
                # Validation for OPENAI_API_KEY is in config.py (checks if base_url is also missing)

                openai_kwargs = {
                    "model_name": self.config.LLM_MODEL,
                    "temperature": self.config.LLM_TEMPERATURE,
                    "max_tokens": self.config.LLM_MAX_TOKENS
                }
                if self.config.OPENAI_API_KEY: # Only pass API key if it's set
                    openai_kwargs["api_key"] = self.config.OPENAI_API_KEY

                if self.config.OPENAI_API_BASE_URL:
                    openai_kwargs["base_url"] = self.config.OPENAI_API_BASE_URL

                self.llm = ChatOpenAI(**openai_kwargs)
                log_msg = (
                    f"Initialized OpenAI LLM Client with model: {self.config.LLM_MODEL}, "
                    f"temp: {self.config.LLM_TEMPERATURE}, max_tokens: {self.config.LLM_MAX_TOKENS}"
                )
                if self.config.OPENAI_API_BASE_URL:
                    log_msg += f", base_url: {self.config.OPENAI_API_BASE_URL}"
                logger.info(log_msg)

            except ImportError:
                logger.error("langchain_openai is not installed. Please install it.")
                raise
            except Exception as e:
                logger.error(f"Error initializing ChatOpenAI: {e}", exc_info=True)
                raise

        elif self.config.LLM_PROVIDER == "azure_openai":
            try:
                from langchain_openai import AzureChatOpenAI # Import for Azure
                # Validation for Azure keys/config is in config.py
                azure_kwargs = {
                    "azure_endpoint": self.config.AZURE_OPENAI_ENDPOINT,
                    "api_key": self.config.AZURE_OPENAI_API_KEY,
                    "api_version": self.config.AZURE_OPENAI_API_VERSION,
                    "azure_deployment": self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "temperature": self.config.LLM_TEMPERATURE,
                    "max_tokens": self.config.LLM_MAX_TOKENS
                }
                self.llm = AzureChatOpenAI(**azure_kwargs)
                logger.info(
                    f"Initialized Azure OpenAI Client with deployment: {self.config.AZURE_OPENAI_DEPLOYMENT_NAME}, "
                    f"endpoint: {self.config.AZURE_OPENAI_ENDPOINT}, temp: {self.config.LLM_TEMPERATURE}, "
                    f"max_tokens: {self.config.LLM_MAX_TOKENS}"
                )
            except ImportError:
                logger.error("langchain_openai is not installed for Azure. Please install it.")
                raise
            except Exception as e: # Catch any other potential errors during init
                logger.error(f"Error initializing AzureChatOpenAI: {e}", exc_info=True)
                raise

        elif self.config.LLM_PROVIDER == "ollama":
            try:
                # from langchain_community.llms import Ollama # Old import
                from langchain_ollama import OllamaLLM # New import

                ollama_kwargs = {
                    "model": self.config.LLM_MODEL,
                    "num_predict": self.config.LLM_MAX_TOKENS  # Set max tokens at initialization
                }
                if hasattr(self.config, "OLLAMA_BASE_URL") and self.config.OLLAMA_BASE_URL:
                    ollama_kwargs["base_url"] = self.config.OLLAMA_BASE_URL

                # Example for temperature, if it were in config:
                # if hasattr(self.config, 'LLM_TEMPERATURE'):
                #    ollama_kwargs['temperature'] = self.config.LLM_TEMPERATURE

                self.llm = OllamaLLM(**ollama_kwargs) # Use new class name
                logger.info(f"Initialized Ollama LLM Client (OllamaLLM) with model: {self.config.LLM_MODEL}, base_url: {ollama_kwargs.get('base_url', 'default')}, num_predict: {self.config.LLM_MAX_TOKENS}")
            except ImportError:
                logger.error("langchain_ollama is not installed. Please install it. You may also need langchain_community for other parts if not already installed.")
                raise
        elif self.config.LLM_PROVIDER == "placeholder_llm": # A default that doesn't require keys
            logger.info("Initialized Placeholder LLM Client.")
            self.llm = "PlaceholderLLMInstance"
        elif self.config.LLM_PROVIDER == "default_llm_provider": # From default config
             logger.info(f"LLM Provider set to '{self.config.LLM_PROVIDER}'. This is a placeholder and will simulate LLM calls.")
             self.llm = "PlaceholderLLMInstance"
        else:
            # Potentially, user has set a provider, but we don't have logic for it yet
            logger.info(f"LLM Provider is '{self.config.LLM_PROVIDER}'. Advanced configuration needed. Using placeholder.")
            self.llm = "PlaceholderLLMInstance"
        logger.debug(f"[LLMClient __init__] self.llm is: '{self.llm}' (Type: {type(self.llm)})")


    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        logger.debug(f"[LLMClient generate_text] self.llm is: '{self.llm}' (Type: {type(self.llm)})")
        if self.llm == "PlaceholderLLMInstance":
            # Simulate LLM response for placeholder
            logger.debug(f"LLMClient (Placeholder) generating text for prompt (first 80 chars): '{prompt[:80]}...'")
            if "generate a concise search query" in prompt.lower():
                response = f"Simulated search query for: {prompt.split(':')[-1].strip()}"
                logger.debug(f"Placeholder LLM generated search query: {response}")
                return response
            elif "summarize the following information" in prompt.lower():
                response = "This is a simulated summary of the provided information."
                logger.debug(f"Placeholder LLM generated summary: {response}")
                return response
            elif "identify key knowledge gaps" in prompt.lower():
                # Simulate reflection: sometimes generate new query, sometimes 'None'
                if "loop 1" in prompt.lower() or "loop 2" in prompt.lower(): # crude way to vary output for testing
                     response = "Simulated refined query based on reflection."
                else: # Simulate ending condition
                     response = "None"
                logger.debug(f"Placeholder LLM generated reflection query: {response}")
                return response

            default_response = f"Simulated LLM response to: {prompt}"
            logger.debug(f"Placeholder LLM generated default response: {default_response}")
            return default_response

        # OpenAI / Azure OpenAI LLM Call (uses .invoke and expects AIMessage with .content)
        if self.config.LLM_PROVIDER == "openai" or self.config.LLM_PROVIDER == "azure_openai":
            provider_name = "OpenAI" if self.config.LLM_PROVIDER == "openai" else "Azure OpenAI"
            model_identifier = self.config.LLM_MODEL if self.config.LLM_PROVIDER == "openai" else self.config.AZURE_OPENAI_DEPLOYMENT_NAME

            # The 'temperature' parameter passed to generate_text is currently ignored for OpenAI/Azure,
            # as it's set during __init__. If per-call temperature is needed, this logic would need adjustment.
            # However, the original generate_text signature had a temperature arg, so we log if it's different from config.
            if temperature != self.config.LLM_TEMPERATURE: # Log if per-call temp differs from init temp
                 logger.warning(f"Call-time temperature {temperature} differs from init temperature {self.config.LLM_TEMPERATURE} for {provider_name}. Using init temperature.")

            logger.info(f"LLMClient generating text with {provider_name} (model/deployment: {model_identifier}) for prompt: '{prompt[:80]}...'")
            try:
                # Temperature and max_tokens are set at initialization
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    if response.content is None or response.content.strip() == "":
                        logger.warning(f"{provider_name} LLM returned None or empty content for prompt: '{prompt[:80]}...'")
                        return ""
                    logger.debug(f"{provider_name} LLM response content: {response.content[:200]}...")
                    return response.content
                else:
                    logger.warning(f"{provider_name} response object does not have 'content' attribute. Response: {str(response)[:200]}")
                    return str(response)
            except Exception as e:
                logger.error(f"Error during {provider_name} LLM invoke: {e}", exc_info=True)
                raise

        # Ollama LLM Call (uses .invoke and returns string directly)
        elif self.config.LLM_PROVIDER == "ollama":
            logger.info(f"LLMClient generating text with Ollama (model: {self.config.LLM_MODEL}, max_tokens via num_predict at init) for prompt: '{prompt[:80]}...'")

            num_retries = self.config.OLLAMA_NUM_RETRIES
            retry_delay = self.config.OLLAMA_RETRY_DELAY_SECONDS
            last_exception = None # To store the exception from the last attempt

            for attempt in range(num_retries + 1): # Initial attempt + num_retries
                try:
                    logger.info(f"Ollama attempt {attempt + 1}/{num_retries + 1} for prompt: '{prompt[:80]}...'")
                    response = self.llm.invoke(prompt) # self.llm is OllamaLLM instance

                    if response is None or response.strip() == "":
                        logger.warning(f"Ollama LLM returned None or empty string on attempt {attempt + 1} for prompt: '{prompt[:80]}...'")
                        # For empty/None responses, we might not want to retry unless it's a sign of a recoverable issue.
                        # For now, let's treat it as a valid (but empty) response and return, not retry.
                        # If retrying on empty is desired, this logic would change.
                        return ""

                    logger.debug(f"Ollama LLM response on attempt {attempt + 1}: {response[:200]}...")
                    return response # Success, exit loop and method

                except ResponseError as e: # Catch specific Ollama client errors (like EOF, connection refused)
                    last_exception = e
                    logger.warning(f"Ollama API ResponseError on attempt {attempt + 1}/{num_retries + 1}: {e}. Retrying in {retry_delay}s...")
                    if attempt < num_retries:
                        time.sleep(retry_delay)
                    else: # Last attempt
                        logger.error(f"Ollama API error after {num_retries + 1} attempts: {e}", exc_info=True)
                        raise # Re-raise the last caught ResponseError
                except Exception as e: # Catch other potential exceptions from invoke or within the try
                    last_exception = e
                    logger.warning(f"Generic exception on Ollama attempt {attempt + 1}/{num_retries + 1}: {e}. Retrying in {retry_delay}s...")
                    if attempt < num_retries:
                        time.sleep(retry_delay)
                    else: # Last attempt
                        logger.error(f"Failed after {num_retries + 1} attempts with generic exception: {e}", exc_info=True)
                        raise # Re-raise the last caught generic exception

            # This part should ideally not be reached if exceptions are re-raised on the last attempt.
            # But as a fallback, if the loop completes without returning or raising:
            logger.error("Ollama generation failed after all retries without specific exception re-raised.")
            if last_exception: # Should have been set if retries occurred
                 raise last_exception # Re-raise the very last exception encountered
            return "Error: Ollama failed to generate text after multiple retries." # Fallback error message

        logger.error(f"LLM provider '{self.config.LLM_PROVIDER}' not fully implemented for text generation or placeholder used.")
        # Fallback for placeholder or unimplemented providers if not caught by placeholder logic
        if self.llm == "PlaceholderLLMInstance": # Should have been handled above
             return f"Simulated LLM response to: {prompt}"
        raise NotImplementedError(f"Actual LLM client logic for provider '{self.config.LLM_PROVIDER}' is not implemented yet.")

import time
from ollama import ResponseError

# Example Usage (for testing this module)
if __name__ == "__main__":
    # Basic logging for example usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing LLMClient...")
    try:
        # Create a dummy Configuration object for testing
        class MockConfiguration:
            LLM_PROVIDER = "placeholder_llm"
            LLM_MODEL = "test_model"
            LLM_MAX_TOKENS = 256 # Example value for testing
            OPENAI_API_KEY = None # Not needed for placeholder
            LOG_LEVEL = "DEBUG"

        mock_config = MockConfiguration()
        llm_client = LLMClient(config=mock_config)

        prompt1 = "Generate a concise search query for the topic: Quantum Computing"
        response1 = llm_client.generate_text(prompt1)
        logger.info(f"Prompt 1: {prompt1}\nResponse 1: {response1}\n")

        prompt2 = "Summarize the following information relevant to 'AI ethics': ..."
        response2 = llm_client.generate_text(prompt2)
        logger.info(f"Prompt 2: {prompt2}\nResponse 2: {response2}\n")

        prompt3 = "Current research topic: Cats\nAccumulated summary so far:\nCats are fluffy.\n\nBased on the summary, identify key knowledge gaps... Generate a new, specific search query... This is for loop 1."
        response3 = llm_client.generate_text(prompt3)
        logger.info(f"Prompt 3 (Reflection): {prompt3}\nResponse 3: {response3}\n")

        prompt4 = "Current research topic: Dogs\nAccumulated summary so far:\nDogs are loyal.\n\nBased on the summary, identify key knowledge gaps... Generate a new, specific search query... This is for loop 3."
        response4 = llm_client.generate_text(prompt4)
        logger.info(f"Prompt 4 (Reflection end): {prompt4}\nResponse 4: {response4}\n")


    except Exception as e:
        logger.error(f"Error in LLMClient example: {e}", exc_info=True)