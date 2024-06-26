
import json
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder

# Function to generate prompts
def generate_prompts(domain, model_path, file_path_2):
    with open(model_path, 'r') as file:
        data = json.load(file)
    with open(file_path_2, 'r') as file_2:
        search_history = json.load(file_2)

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

    llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                  api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                                  token=Secret.from_token("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"))

    document_store = InMemoryDocumentStore()
    document_store.write_documents([
        Document(content=f"""
        ATTRIBUTE_TYPES = {ATTRIBUTE_TYPES}

        ENTITY_TYPES = {ENTITY_TYPES}
        """),
        Document(content=f"""

Uses the following(but not limited to) attribute and entity types:
Attribute Type:
Continuous: To measure growth metrics (e.g., revenue, user base).
Ordered: To rank growth rates.
Entity Type:
Fact: To calculate growth based on factual data over time.
Key-Value Conditions: aggregationMethods, rowCount

Query semantics: <Action> <Subject> <Time Frame> <Comparison> <Conditions>
Here, <Subject> must be taken from {ENTITY_TYPES['Fact']}.
Rest should be taken from {ATTRIBUTE_TYPES['Continuous']} and {ATTRIBUTE_TYPES['Ordered']}.
"""),
         

            Document(content=f"""
        Previous Search History:
        {json.dumps(search_history, indent=4)}
        """)
    ])

    # search_history_str = "\n".join(search_history)

    prompt_template = """
    Given these documents, the example graphical schema, and the search history, answer the question.
    NOTE: The data in the schema is sample data only for your reference.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Schema: {graph.schema}
    Search History: {search_history}
    Question: {{question}}
    Answer:
    """

    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    question = f"Based on {domain} data, suggest 10 natural language search prompts to get meaningful insights from the data."

    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    # Print the raw LLM response for debugging
    res = results["llm"]["replies"]
    print("LLM Response:", res)  # Debug logging

    # Process the LLM response to extract only the first 10 prompts
    raw_prompts = res[0].split('\n')
    
    # Extract the first 10 valid prompts
    initial_prompts = []
    for line in raw_prompts:
        line = line.strip()
        if line and line[0].isdigit() and line[1] == '.':
            prompt = line.split('.', 1)[1].strip().strip('"')
            if prompt:
                initial_prompts.append(prompt)
            if len(initial_prompts) == 10:
                break

    attribute_analysis_question = {
        "prompts": initial_prompts,
        "attribute_types": ATTRIBUTE_TYPES
    }

    question = f"""Identify the attributes being referenced in each prompt for the below prompts and attribute types:
    {attribute_analysis_question}
    """
    
    # Second LLM call to analyze attributes
    attribute_analysis_prompt_template = """
    Given the following prompts and attribute types, answer the question.
    Prompts:
    {% for prompt in prompts %}
        {{ prompt }}
    {% endfor %}
    Attribute Types: {{ attribute_types }}
    Question: {{question}}
    Answer in the format:
    Prompt: <prompt>
    Attributes: <comma-separated list of attributes>
    """

    prompt_builder.template =  attribute_analysis_prompt_template



    attribute_results = rag_pipeline.run(
        {
             "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    # Print the raw LLM response for debugging
    attribute_res = attribute_results["llm"]["replies"]
    print("Attribute Analysis LLM Response:", attribute_res)  # Debug logging

    # Extract the attributes for each prompt
    final_prompts_with_attributes = []
    for line in attribute_res[0].split('\n'):
        if line.startswith("Prompt:"):
            prompt_text = line.split("Prompt:")[1].strip()
        if line.startswith("Attributes:"):
            attributes_text = line.split("Attributes:")[1].strip()
            attributes_list = attributes_text.split(',')
            final_prompts_with_attributes.append({"prompt": prompt_text, "attributes": attributes_list})

    print("Final Prompts with Attributes:", final_prompts_with_attributes)  # Debug logging

    return final_prompts_with_attributes

# Test call for debugging (uncomment if you want to run this directly)
# result = generate_prompts("customer", "path_to_your_model.json", "path_to_your_search_history.json")
# print(result)
