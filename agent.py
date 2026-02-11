from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_agent(df, similarity_matrix, rules, hybrid_recommend):
    """
    Creates a movie recommendation agent using Ollama
    """
    
    def recommend_tool(movie_name: str) -> str:
        """Recommends movies based on the input movie name"""
        try:
            movies = hybrid_recommend(movie_name, df, similarity_matrix, rules)

            if movies[0] == "Movie not found in dataset":
                return f"Sorry, '{movie_name}' was not found in the dataset."

            response = f"Top 4 movies similar to '{movie_name}':\n\n"

            for i, movie in enumerate(movies, 1):
                response += f"{i}. {movie}\n"

            return response
        except Exception as e:
            return f"Error: {str(e)}"

    # Initialize Ollama LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful movie recommendation assistant. 
When the user asks for movie recommendations, extract the movie name and use it to provide recommendations.
Always be friendly and helpful."""),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm | output_parser
    
    def agent(user_input: str) -> str:
        """
        Process user input and return movie recommendations
        
        Args:
            user_input: User's message
            
        Returns:
            Formatted recommendation response
        """
        
        # Step 1: Extract movie name using LLM
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the movie name from the user's message. Return ONLY the movie name, nothing else. If no movie is mentioned, return 'NONE'."),
            ("human", "{input}")
        ])
        
        extraction_chain = extraction_prompt | llm | output_parser
        movie_name = extraction_chain.invoke({"input": user_input}).strip().strip('"').strip("'")
        
        if movie_name == "NONE" or not movie_name:
            return "Please tell me which movie you'd like recommendations for!"
        
        # Step 2: Get recommendations
        recommendations = recommend_tool(movie_name)
        
        # Step 3: Format with LLM for natural response
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly movie recommendation assistant. Present these recommendations in a natural, conversational way."),
            ("human", "User asked about: {movie_name}\n\nRecommendations:\n{recommendations}\n\nPresent this naturally.")
        ])
        
        format_chain = format_prompt | llm | output_parser
        
        final_response = format_chain.invoke({
            "movie_name": movie_name,
            "recommendations": recommendations
        })
        
        return final_response
    
    return agent


# Simpler version without extra formatting
def create_simple_agent(df, similarity_matrix, rules, hybrid_recommend):
    """
    Creates a simpler movie recommendation agent
    """
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    output_parser = StrOutputParser()
    
    # Extraction chain
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the movie name from the user's message. Return ONLY the movie name. If no movie is mentioned, return 'NONE'."),
        ("human", "{input}")
    ])
    
    extraction_chain = extraction_prompt | llm | output_parser
    
    def agent(user_input: str) -> str:
        # Extract movie name
        movie_name = extraction_chain.invoke({"input": user_input}).strip().strip('"').strip("'")
        
        if movie_name == "NONE" or not movie_name:
            return "Please specify a movie name to get recommendations!"
        
        # Get recommendations
        try:
            movies = hybrid_recommend(movie_name, df, similarity_matrix, rules)
            
            if movies and movies[0] == "Movie not found in dataset":
                return f"Sorry, I couldn't find '{movie_name}' in the database."
            
            result = f"🎬 Movies similar to '{movie_name}':\n\n"
            for i, movie in enumerate(movies, 1):
                result += f"{i}. {movie}\n"
            
            return result
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    return agent