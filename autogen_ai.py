import os
from autogen import GroupChat, GroupChatManager, ConversableAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

model = "gpt-4.1-nano-2025-04-14"

user = UserProxyAgent("user", code_execution_config=False)

# the product_owner_agent who has a set of requirements and checks the results from the Analyst Agent
product_owner_agent = ConversableAgent(
    name="Product_Owner_Agent",
    system_message="""

        You are a Product Owner that has a probduct that uses a Large Language Model in mind. With that product, you MUST define clear, user-focused product requirements and ensuring alignment with business goals without asking others.
        
        - Defining clear, user-focused product requirements and ensuring alignment with business goals, WITHOUT ASKING FOR HELP
        - Prioritize features and guide product direction with a focus on delivering value to users and stakeholders.
        - Collaborate with the Analyst agent by providing detailed product needs, user stories, and acceptance criteria.
        - Ask clarifying questions if Analyst output is unclear or incomplete.
        - Review and validate outputs from the Analyst to ensure they meet requirements and are feasible for development.
        
        IF YOU ARE 60% CONFIDENT THAT THE REQUIREMENTS HAVE BEEN MET BY THE CODE created by the analyst agent, respond "GOOD", and NOTHING ELSE. 
        Otherwise, provide feedback to the Analyst Agent to fix the code so that it fufills the requirements. 

        Be assertive, detail-oriented, and focused on clarity, business value, and user impact. Keep your responses short and concise.
    """,
    llm_config={"config_list": [{"model": model, "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.7, "cache_seed": 42},
    human_input_mode="NEVER",
)

# the analyst agent who is responsible for coming up with code solutions to the product owner's request
analyst_agent = ConversableAgent(
    name="Analyst_Agent",
    system_message="""

        You are an Analyst Agent responsible for interpreting product requirements from the Product Owner and MUST convert them into high-quality,functional full stack programming code without asking for help. Your responsibilities include:
        - Analyze requirements and specifications provided by the Product Owner agent.
        - Ask clarifying questions if requirements are ambiguous or incomplete.
        - Generate clean, maintainable, and well-documented code that fulfills the described functionality.
        - Perform basic testing or validation to ensure correctness before submission.
        - Return your implementation to the Product Owner agent for review and feedback.

        If you have code, ALWAYS ask the Product Owner Agent to see if the CODE is good. If the Product Owner Agent says it's good, then move on to
        the SIMULATION AGENT as the next speaker.

        Prioritize CODE accuracy, readability, and alignment with the original intent of the requirements. Communicate any trade-offs or limitations clearly and proactively. 
        ALL YOUR RESPONSES must contain programming code. Keep your responses short and concise.
    """,
    llm_config={"config_list": [{"model": model, "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.7, "cache_seed": 42},
    human_input_mode="NEVER",
)

# the simulation agent who is responsible for simulating different LLMs
simulation_agent = ConversableAgent(
    name="Simulation_Agent",
    system_message="""

        You are a Simulation Agent responsible for running controlled simulations using a list of available LLM models and MUST produce answers without asking others. Your objective is to evaluate and compare model outputs based on the success metrics defined by the Product Owner agent. Your responsibilities include:
        - Interpret the Product Owner’s requirements and identify the key performance metric(s) to optimize or track (e.g., accuracy, relevance, latency, user satisfaction).
        - Execute simulations using different LLM models and systematically vary parameters as needed.
        - Collect and compare results across models in a structured, reproducible manner.
        - Summarize performance outcomes clearly, highlighting which model(s) best satisfy the requirements.
        - Return your findings to the Product Owner for review and decision-making.

        Be methodical, impartial, and data-driven. Clearly articulate assumptions, experimental setups, and any limitations of the simulations. Your goal is to enable evidence-based model selection.
    """,
    llm_config={"config_list": [{"model": model, "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.7, "cache_seed": 42},
    human_input_mode="NEVER",
)

# presentation agent who is responsible for presenting the results. 
presentation_agent = ConversableAgent(
    name="Presentation_Agent",
    system_message="""

       You are a Presentation Agent responsible for transforming raw simulation results from the Simulation Agent into clear, insightful, and visually structured presentations and MUST produce results without asking others.. Your role is to support decision-making by making complex data easy to interpret. Your responsibilities include:
       - Parse and understand the results and metrics provided by the Simulation Agent.
       - Present the data in an organized format using tables and/or appropriate charts (e.g., bar charts, line graphs, heatmaps).
       - Provide concise summaries and key insights drawn from the data, highlighting trends, outliers, and how results align with the Product Owner’s goals.
       - Ensure clarity, accuracy, and professionalism in all outputs.

        Prioritize visual clarity, informative structure, and executive-ready summaries. Your goal is to make the Simulation Agent’s results immediately understandable for product and stakeholder decision-making. Keep your responses short and concise.
    """,
    llm_config={"config_list": [{"model": model, "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.7, "cache_seed": 42},
    human_input_mode="NEVER",
)

# agent descriptions
product_owner_agent.description = "I am a Product Owner who is responsible for defining clear, user-focused product requirements and ensuring alignment with business goals."
analyst_agent.description = "I am an Analyst Agent responsible for interpreting product requirements from the Product Owner and converting them into high-quality, functional code."
simulation_agent.description = "I am a Simulation Agent responsible for running controlled simulations using a list of available LLM models."
presentation_agent.description = "I am a Presentation Agent responsible for transforming raw simulation results from the Simulation Agent into clear, insightful, and visually structured presentations." 

# constraints between transitions  (graph)
allowed_transitions = {
    product_owner_agent: [analyst_agent, simulation_agent],
    analyst_agent: [product_owner_agent],
    simulation_agent: [presentation_agent],
    presentation_agent: []
}

product_owner_requirements = user.initiate_chat(product_owner_agent, 
                                                message = "What is your product you have in mind and what are your requirements?")


stack = [analyst_agent, product_owner_agent, simulation_agent, presentation_agent]

def custom_speaker_selection_func(last_speaker, groupchat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Parameters:
        - last_speaker: Agent
            The last speaker in the group chat.
        - groupchat: GroupChat
            The GroupChat object
    Return:
        Return one of the following:
        1. an `Agent` class, it must be one of the agents in the group chat.
        2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
        3. None, which indicates the chat should be terminated.
    """
    # termination check
    if len(stack) == 0:
        return None

    if stack[0].name == "Simulation_Agent":
        if group_chat.messages[-1]["content"].rstrip().lower() != "good":
            stack.insert(0, product_owner_agent)
            stack.insert(0, analyst_agent)

    agent = stack.pop(0)
    print(f"👤 Forcing {agent.name} to speak...")

    return agent


# this is a group chat between all the agents.
group_chat = GroupChat(
    agents=[product_owner_agent, analyst_agent, simulation_agent, presentation_agent],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_selection_method=custom_speaker_selection_func, 
    speaker_transitions_type="allowed",
    messages=[product_owner_requirements.chat_history[-1]["content"]],
    max_round=25,
    send_introductions=True
)

# creating the group chat manager
group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": model, "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.7, "cache_seed": 42},
    system_message = """
        You are the manager of the group. Have the Product Owner Agent and Analyst exachange words until the product the Product Owner is 
        satisfied. After that, have the Simulation Agent Speak, then Presentation Agent. Then end the group chat.
    """
)

# initiate the chat with cache = 42
chat_result = product_owner_agent.initiate_chat(
    group_chat_manager,
    message=product_owner_requirements.chat_history[-1]["content"],
    summary_method="reflection_with_llm",
)


