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

    question = f"Based on {domain} data suggest 10 simple natural language search prompts to get meaningful insights from the data. Use the search history file added as examples. Ignore all cached data"

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
    final_prompts = []
    for line in raw_prompts:
        line = line.strip()
        if line and line[0].isdigit() and line[1] == '.':
            prompt = line.split('.', 1)[1].strip().strip('"')
            if prompt:
                attributes_found = []
                for attr_type, attr_list in ATTRIBUTE_TYPES.items():
                    for attr in attr_list:
                        if attr.lower() in prompt.lower():
                            attributes_found.append(attr)
                final_prompts.append({"prompt": prompt, "attributes": list(attributes_found)})
            if len(final_prompts) == 10:
                break

    print("Extracted Prompts:", final_prompts)  # Debug logging

    return final_prompts

# Test call for debugging (uncomment if you want to run this directly)
# result = generate_prompts("customer", "path_to_your_model.json", "path_to_your_search_history.json")
# print(result)

