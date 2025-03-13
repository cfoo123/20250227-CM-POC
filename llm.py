from openai import OpenAI
import pandas as pd
import asyncio
import json
from openai import AsyncOpenAI
from tqdm import tqdm
import os

openai_api = os.getenv("OPENAI_API_KEY")

def api_client():
    api_key = openai_api

    client = AsyncOpenAI(
        base_url="https://staging-litellm.moneylion.io",
        api_key=f"{api_key}",
    )
    return api_key, client

api_key, client = api_client()

def credit_improvement_prompts(score):
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
    Analyze the following credit score data and produce an output in valid JSON format (using the keys defined in the system prompt). Incorporate an empathetic, encouraging tone while providing a detailed breakdown of the credit situation and clear recommendations:

    "snap_shot__open_accounts": {score["snap_shot__open_accounts"]},
    "snap_shot__total_closed_accounts": {score["snap_shot__total_closed_accounts"]},
    "snap_shot__closed_accounts": {score["snap_shot__closed_accounts"]},
    "snap_shot__total_accounts": {score["snap_shot__total_accounts"]},
    "snap_shot__delinquent_accounts": {score["snap_shot__delinquent_accounts"]},
    "snap_shot__derogatory_accounts": {score["snap_shot__derogatory_accounts"]},
    "snap_shot__total_balances": {score["snap_shot__total_balances"]},
    "snap_shot__total_monthly_payments": {score["snap_shot__total_monthly_payments"]},
    "snap_shot__number_of_inquiries": {score["snap_shot__number_of_inquiries"]},
    "snap_shot__total_public_records": {score["snap_shot__total_public_records"]},
    "snap_shot__available_credit": {score["snap_shot__available_credit"]},
    "snap_shot__utilization": {score["snap_shot__utilization"]},
    "snap_shot__date_of_oldest_trade": {score["snap_shot__date_of_oldest_trade"]},
    "snap_shot__age_of_credit": {score["snap_shot__age_of_credit"]},
    "snap_shot__total_open_installment_accounts": {score["snap_shot__total_open_installment_accounts"]},
    "snap_shot__total_open_collection_accounts": {score["snap_shot__total_open_collection_accounts"]},
    "snap_shot__total_open_mortgage_accounts": {score["snap_shot__total_open_mortgage_accounts"]},
    "snap_shot__total_open_revolving_accounts": {score["snap_shot__total_open_revolving_accounts"]},
    "snap_shot__total_open_other_accounts": {score["snap_shot__total_open_other_accounts"]},
    "snap_shot__balance_open_installment_accounts": {score["snap_shot__balance_open_installment_accounts"]},
    "snap_shot__balance_open_collection_accounts": {score["snap_shot__balance_open_collection_accounts"]},
    "snap_shot__balance_open_mortgage_accounts": {score["snap_shot__balance_open_mortgage_accounts"]},
    "snap_shot__balance_open_revolving_accounts": {score["snap_shot__balance_open_revolving_accounts"]},
    "snap_shot__balance_open_other_accounts": {score["snap_shot__balance_open_other_accounts"]},
    "snap_shot__on_time_payment_percentage": {score["snap_shot__on_time_payment_percentage"]},
    "snap_shot__late_payment_percentage": {score["snap_shot__late_payment_percentage"]},
    "thin_file__description": {score["thin_file__description"]},
    "vantage_score3": {score["vantage_score3"]},
    "vantage_score_category": {score["vantage_score_category"]}

    Please ensure your final answer is supportive, understandable, and presented solely as a JSON object with the specified structure.
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

async def get_credit_improvement(score_list: list, analysis_type: str = 'credit_improvement', model: str = 'gpt-4o-mini') -> dict:
    
    # List to hold all asynchronous requests
    # Map hashed comment_id to temporary numbers (1-10)
    requests, loop_count = [],0

    # Loop over each chunk of 10 comments
    for score in tqdm(score_list):
        # Read system and user messages based on analysis type
        if analysis_type == 'credit_improvement':
            system_prompt, user_prompt = credit_improvement_prompts(score)
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

def calculate_accuracy_metrics(responses, df_input):
    # Convert responses into a list of dictionaries
    response_dicts = []
    for i in range(len(responses)):
        response_i = json.loads(responses[i].choices[0].message.content)
        response_i['user_id'] = df_input[i]['user_id']
        response_dicts.append(response_i)

    # Initialize lists to store evaluation results
    evaluate_score = []
    evaluate_delinquent = []
    evaluate_derogatory = []
    evaluate_late_payment = []

    # Perform evaluation for each response
    for i, response in enumerate(response_dicts):
        user_data = df_input[i]
        score_health = response['Score Health']  # assume it's a string
        dragging_down = ' '.join(response["What's dragging down your score?"]).lower()
        
        # Evaluate if the correct VantageScore is mentioned in 'Score Health'
        evaluate_score.append(str(user_data['vantage_score3']) in score_health)
        
        # Evaluate if delinquent accounts are correctly mentioned
        evaluate_delinquent.append(
            not (user_data['snap_shot__delinquent_accounts'] > 0 and 'delinquent' not in dragging_down)
        )
        
        # Evaluate if derogatory accounts are correctly mentioned
        evaluate_derogatory.append(
            not (user_data['snap_shot__derogatory_accounts'] > 0 and 'derogatory' not in dragging_down)
        )
        
        # Convert late payment percentage to float and evaluate if it's correctly mentioned
        late_payment_percentage_real = float(user_data['snap_shot__late_payment_percentage'].rstrip('%')) / 100.0
        evaluate_late_payment.append(
            not (late_payment_percentage_real > 0 and 'late payment' not in dragging_down)
        )

    # Calculate the mean accuracy for each metric
    accuracy_metrics = {
        'Metric': [
            'Vantage Score Accuracy',
            'Delinquent Accounts Accuracy',
            'Derogatory Accounts Accuracy',
            'Late Payment Accuracy'
        ],
        'Accuracy': [
            sum(evaluate_score) / len(evaluate_score),
            sum(evaluate_delinquent) / len(evaluate_delinquent),
            sum(evaluate_derogatory) / len(evaluate_derogatory),
            sum(evaluate_late_payment) / len(evaluate_late_payment)
        ]
    }

    # Return the accuracy metrics as a JSON string
    return json.dumps(accuracy_metrics, indent=4)
