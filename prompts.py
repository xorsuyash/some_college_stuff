class EmailPromptTemplate:

    def generate_synthetic_email_response(email):


       return f"""You are trained to generate professional, polite, and concise email replies. Given an email, generate a short and relevant response that acknowledges the content while keeping it general and adaptable to different contexts. Ensure the tone is neutral, courteous, and professional.

        ##Instructions
        1. Keep the response between 1 to 3 sentences.
        2. If clarification is needed, express willingness to assist further.
        3. Avoid unnecessary details while ensuring politeness.
        4. Do not include subject in response only include body of email response that you are generating.
        4. Use placeholders like "name" and "specific detail" where customization is expected.
        5. Return the reponse in clear json format. 

        ##email:
        {email}
        """
    def evaluate_response(email,generated_response):
        pass 
    
    def evolve_input(email,generated_response,score,feedback):

        return f"""
                You are an AI trained to improve email replies based on evaluation feedback. Given an email, an initial response, an average score, and feedback, your task is to rewrite the reply to enhance clarity, relevance, politeness, and adaptability while keeping it concise.

                Instructions:
                1. Analyze the given input: Identify areas for improvement based on the feedback.
                2. Rewrite the response to address the feedback while ensuring:
                - Relevance: Fully responds to the email's intent.
                - Conciseness: Brief and to the point.
                - Politeness: Uses a professional and courteous tone.
                - Adaptability: Includes placeholders (e.g., "name") for easy customization.
                3. Preserve the original meaning but refine structure, tone, and clarity.

                Input Format:
                Email: "{email}"
                Generated Response: "{generated_response}"
                Average Score: {score}
                Feedback: "{feedback}"

                Expected Output Format:
                Refined Response:

                """