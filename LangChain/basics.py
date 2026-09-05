# ------------------------------------------------------------------------------------------
# 1. MODELS - The Reasoning Engine
# ------------------------------------------------------------------------------------------
# The "model" is the LLM itself - the part that actually thinks.

# 'init_chat_model' gives ONE interface to every provider i.e. LangChain lets 
# you talk to different AI providers using roughly the same Python code.

from langchain.chat_models import init_chat_model

model = init_chat_model("minimax-m3:cloud", model_provider="ollama")

# .invoke() is the universal "run it" method across LangChain.
# .content = just the text of the reply
print(model.invoke("What is LangChain in one sentence?").content)

# BELOW IS ANOTHER WAY TO INITIALIZE THE SAME MODEL.

# from langchain_ollama import ChatOllama
#
# model = ChatOllama(
#     model="minimax-m3:cloud"
# )
#
# response = model.invoke("What is LangChain in one sentence?")
#
# print(response.content)


# ------------------------------------------------------------------------------------------
# 2. PROMPT TEMPLATES - Steering the Model
# ------------------------------------------------------------------------------------------
# A prompt template is a reusable sentence with blanks ({placeholders}) you
# fill in later - write the wording once, reuse it many times.

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# PromptTemplate = a single plain-text string with blanks.
simple_template = PromptTemplate(
    input_variables = ["topic"],
    template = "Explain {topic} to a complete beginner in 2-3 sentences."
)

# .format() fills the blank and returns the finished text.
formatted = simple_template.format(topic="AI agents")
print(" === Formatted Prompt === ")
print(formatted)
print()

# ChatPromptTemplate = built from ROLES (system / human), which is how chat
# models expect their input. "system" sets behavior, "human" is the user turn.
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding tutor. Keep answers short and clear."),
    ("human", "Explain {concept} with a simple Python example."),
])

# .format_messages() fills the blanks and returns a LIST of messages.
messages = chat_template.format_messages(concept="list comprehension")
# {concept} in the template is the empty blank, and concept="list comprehension" 
# is you handing it the word that goes in the blank.
print(" === Chat Messages === ")
for msg in messages:
    print(f" [{msg.type}]: {msg.content[:80]} ... ")
    print()


# ------------------------------------------------------------------------------------------
# 3. CHAINS - Connecting the Pieces with LCEL
# ------------------------------------------------------------------------------------------
# The pipe | glues components into a pipeline. Each step's output flows
# into the next, left to right: prompt | model | parser.

from langchain_core.output_parsers import StrOutputParser

# prompt fills the blanks. model answers + parser pulls out clean text.
# StrOutputParser just extracts the plain string from the model's reply
# object, so you don't have to write .content yourself every time.
chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke().
result = chain.invoke({"concept": "for loops"})
print(" === Chain Output === ")
print(result)
print()


# ------------------------------------------------------------------------------------------
# 4. MEMORY - Giving the Model Context
# ------------------------------------------------------------------------------------------
# Models are stateless - they forget everything between calls.
# "Memory" is simply us storing past messages and feeding them back in.

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# A simple in-memory store that holds the conversation.
memory = InMemoryChatMessageHistory()

# Hand-build a short conversation so we have something to "remember".
memory.add_message(HumanMessage(content="My name is Yash and I'm learning LangChain"))
memory.add_message(AIMessage(content="Nice to meet you, Yash! LangChain is a great choice."))
memory.add_message(HumanMessage(content="What tools should I learn first?"))
memory.add_message(AIMessage(content="Start with PromptTemplates and simple chains, then move to tools and agents."))

print(" === Memory Contents === ")
for msg in memory.messages:
    print(f" [{msg.type}]: {msg.content[:80]} ... ")
print()

# The "placeholder" slot is where the stored messages get injected into the
# prompt, so the model can SEE the earlier conversation.
chat_with_memory = ChatPromptTemplate.from_messages ([
    ("system", "You are a helpful tutor. Use the conversation history to personalize your responses."),
    ("placeholder", "{history}"),       # past messages get dropped in here
    ("human", "{question}"),
])

chain_with_memory = chat_with_memory | model | StrOutputParser()

# We pass the stored history in alongside the new question.
# Watch the model correctly recall the name "Yash" - that's "memory".
result = chain_with_memory.invoke({
    "history": memory.messages,
    "question": "What was my name again?"
})
print(" === Memory-Aware Response === ")
print(result)
print()