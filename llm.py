from openai import OpenAI
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm
import os

openai_api = os.getenv("OPENAI_API_KEY")

def api_client():
    api_key = openai_api

    client = OpenAI(
        base_url="https://staging-litellm.moneylion.io",
        api_key=f"{api_key}",
    )
    return api_key, client

api_key, client = api_client()

def credit_improvement_prompts(comments):
    # Construct system message
    system_prompt = """
    You are a warm, empathetic credit analysis with expertise in VantageScore 3.0. You are given credit score data and must produce a concise, user-friendly explanation of the credit situation in valid JSON. 
    The style of your response should be encouraging and supportive, while still communicating important details and actionable steps. 

    Your response must use exactly the following JSON structure and keys (no extra keys or nesting):
    {
    "Score Health": "",
    "What's dragging down your score?": [],
    "What's boosting your score?": [],
    "What might help improve the score?": [],
    "Conclusion": "",
    "Summary": ""
    }

    Important:
    1. Keep the JSON valid—no additional fields or text outside of the JSON object.
    2. Use an empathetic yet clear tone. 
    3. Do not just give a value. Elaborate each point in a concise way
    4. Provide specific details about factors affecting credit score and easy to understand explanation.
    5. Focus more on factors that have greater impact on credit score
    6. in each factor, give the factor values before writing the explanation
    7. Offer actionable recommendations in the "What Might Help Improve the Score" section.
    8. Keep each list entry short but supportive, avoiding overly dry or blunt language, and elaborate each point in a concise way on why is the factor important
    9. Focus on empowering the user to take steps that improve their credit score.
    """
    
    # Construct the prompt to send to OpenAI API
    user_prompt = f"""
        Here's a list of users' comment {comments} commenting on MoneyLion app
        Categorize each comment into its sentiment "positive","neutral", or "negative"
        and output it in a json format

        Expected Output:
        
        {r"{comments:[{'comment': 'comment_xxx','sentiment': 'sentiment_xxx'},{'comment': 'comment_yyy','sentiment': 'sentiment_yyy'}]}"}
        
        ignore double quotes and blackslashes in the comments and do not split 1 comment into multiple comments
        """
    return system_prompt,user_prompt

def get_top5(system_prompt: str, user_prompt: str, model: str = 'gpt-4o-mini'):
    response = client.chat.completions.create(
                messages=[
                    {"role":"system", "content":f"{system_prompt}"},
                    {"role": "user","content": f"{user_prompt}",}
                ],
                model=model, # Specify the model
            # # Define the response format as a JSON object
            response_format={'type':'json_object'},
            )
    return response

async def get_comment_analysis(score_list: list, analysis_type: str, model: str = 'gpt-4o-mini', chunk_size=5) -> dict:
    
    # List to hold all asynchronous requests
    # Map hashed comment_id to temporary numbers (1-10)
    requests, loop_count = [],0

    # Loop over each chunk of 10 comments
    for score in tqdm(score_list):
        # Read system and user messages based on analysis type
        if analysis_type == 'credit_improvement':
            system_prompt, user_prompt = credit_improvement_prompts(chunk)
        else:
            raise ValueError("Invalid analysis type. Choose 'credit_improvement' for this argument")

        # Create an asynchronous request to the OpenAI API for each comment chunk
        chat_completion = client.chat.completions.create(
            messages=[
                {"role":"system", "content":f"{system_prompt}"},
                {"role": "user","content": f"{user_prompt}",}
            ],
            model=model, # Specify the model
        # Define the response format as a JSON object
        response_format={'type':'json_object'},
        )
        
        requests.append(chat_completion)
        loop_count+=1

    # Gather all the asynchronous requests and wait for them to complete
    responses = await asyncio.gather(*requests)

    return responses

