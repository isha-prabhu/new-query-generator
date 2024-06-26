import os
from getpass import getpass
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from langchain import hub
from uuid import uuid4
from langchain.agents import AgentExecutor, create_react_agent

SESSION_ID = str(uuid4())

# HUGGINGFACEHUB_API_TOKEN = getpass()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory




import json
file_path = './aos1005.json'
with open(file_path, 'r') as file:
    data = json.load(file)

entityRelations = {
    "relations": [ ]
}

for rel in data.get("entityRelations"):
    element = {
        "lhsEntityName": rel['lhsEntityName'],
        "rhsEntityName":rel['rhsEntityName'],
    }
    entityRelations['relations'].append(element)

print(entityRelations)

ATTRIBUTE_TYPES = {
    "Continuous": [],
    "Categorical": [],
    "Ordered": [],
    "Boolean": [],
    "Database ID": [],
    "Generic string": [],
    "Timezone": [],
    "Timestamp": [],
}

ENTITY_TYPES = {
    "Fact": [],
    "Dimension": []
}

def classify_attribute(attr):
    logical_data_type = attr.get("logicalDataType")
    if logical_data_type == "CATEGORICAL":
        return "Categorical"
    elif logical_data_type == "DATABASE_ID":
        return "Database ID"
    elif logical_data_type == "CONTINUOUS":
        return "Continuous"
    elif logical_data_type == "BOOLEAN":
        return "Boolean"
    elif logical_data_type == "GENERIC_STRING":
        return "Generic string"
    elif logical_data_type == "TIMEZONE":
        return "Timezone"
    elif logical_data_type == "TIMESTAMP":
        return "Timestamp"
    else:
        return None

for entity in data.get("entityList", []):
    entity_type = entity.get("entityType")
    entity_name = entity.get("canonicalName")

    if entity_type == "Fact" or entity_type == "Dimension":
        ENTITY_TYPES[entity_type].append(entity_name)

    for attribute in entity.get("attributeList", []):
        attr_type = classify_attribute(attribute)
        if attr_type:
            ATTRIBUTE_TYPES[attr_type].append(attribute.get("canonicalName"))

# print("ATTRIBUTE_TYPES =", json.dumps(ATTRIBUTE_TYPES, indent=4))
# print("ENTITY_TYPES =", json.dumps(ENTITY_TYPES, indent=4))

repo_id = "HuggingFaceH4/zephyr-7b-beta"
# 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

llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                    api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                                    token=Secret.from_token("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"))

RESPONSE = {
    "relations": [
        "relationship-for-1st-pair",
        "relationship-for-2nd-pair",
        "relationship-for-3rd-pair",
    ]
}

question = f"""
You are an expert in analysing relationships between different entities. 
Based on the provided list of entities and attributes you need to respond with a JSON object as explained below.
RETURN: {RESPONSE['relations']}, an ordered array of relations where 1st element in this array corresponds to the relationship between 1st pair of entities in the given entityRelations object: {entityRelations['relations']}. 
This format of object must be strictly followed in your responses. 
It is preferred that the relationships can be represented by a single word.
For example,
If an entity pair is:
    "lhsEntityName": "customer",
    "rhsEntityName":"product",
then, the possible relation is "BUYS" because: Customer-BUYS->Product.
Thus, you return ["BUYS"].
NOTE: Only return a single word summarizing the relationship and make sure to close the braces and quotes appropriately.
"""

document_store = InMemoryDocumentStore()
document_store.write_documents([
Document(content=f"""
Following are entities and their attributes with their type.
ATTRIBUTE_TYPES = {ATTRIBUTE_TYPES}

ENTITY_TYPES = {ENTITY_TYPES}
"""),
Document(content=f"""
Following are the possible relationships between entities:
{entityRelations}
"""         
)
])

prompt_template = """
Given these documents, answer the question.

Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

# CYPHER_GENERATION_TEMPLATE = """
# You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
# Convert the user's question based on the schema.

# Schema: {schema}
# Question: {question}
# """

def generate():
    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)

    cypher_pipeline = Pipeline()
    cypher_pipeline.add_component("retriever", retriever)
    print("cy r added")
    cypher_pipeline.add_component("prompt_builder", prompt_builder)
    print("cy pd added")
    cypher_pipeline.add_component("llm", llm)
    print("cy llm added")
    cypher_pipeline.connect("retriever", "prompt_builder.documents")
    cypher_pipeline.connect("prompt_builder", "llm")


    results = cypher_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )
    res = results["llm"]["replies"]
    print("RAG:",type(res[0]))
    print("RAG:",len(res))
    print("RAG:",len(res[0].split(', ')))
    relns = []
    i = 0
    for x in res[0].split(', '):
        if i==0:
           x = x.split('[')[1]
        x=x.removeprefix("'")
        x=x.removesuffix("'")
        relns.append(x)
        i+=1
    print(relns)
    
    return relns
# generate()