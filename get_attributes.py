import json
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder

# Function to classify attributes and use them in the document store
def classify_attributes(model_path):
    with open(model_path, 'r') as file:
        data = json.load(file)
    
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

    return ATTRIBUTE_TYPES, ENTITY_TYPES

# Function to extract attributes used in prompts
def get_attributes(model_path, prompts):
    ATTRIBUTE_TYPES, ENTITY_TYPES = classify_attributes(model_path)

    document_store = InMemoryDocumentStore()
    document_store.write_documents([
        Document(content=f"""
        ATTRIBUTE_TYPES = {ATTRIBUTE_TYPES}

        ENTITY_TYPES = {ENTITY_TYPES}
        """),
        Document(content=f"""
        Uses the following (but not limited to) attribute and entity types:
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
        Example 1  Input :"actual call volume by queue names last month", Output : ("acv","queueid","month_of_year")
        Example 2  Input :"what is the average shrinkage over the last 2 weeks by category names", Output : ("shrinkage","shrinkage_total","category_name")
        """),
    ])

    prompt_template = """
    Given these documents, examples of input-ouput, answer the question.
    NOTE: The data in the schema is sample data only for your reference.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Schema: {graph.schema}
    Question: {{question}}
    Answer:
    """

    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                  api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                                  token=Secret.from_token("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")))

    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    attributes_in_prompts = []

    for prompt in prompts:
        question = f"Analyze the following prompt, understand what it means, and then go thrugh the attribut list to find out the most relevant attributes which can answer the prompt. {prompt}"

        results = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
            }
        )

        res = results["llm"]["replies"][0]
        print("LLM Response:", res)  # Debug logging

        attributes_found = []
        for attr_type, attr_list in ATTRIBUTE_TYPES.items():
            for attr in attr_list:
                if attr.lower() in res.lower():
                    attributes_found.append(attr)
        
        attributes_in_prompts.append({
            "prompt": prompt,
            "attributes": list(set(attributes_found))
        })

    print("Extracted Attributes:", attributes_in_prompts)  # Debug logging

    return attributes_in_prompts

# Example call for testing
# model_path = "path_to_your_model.json"
# prompts = ["Example prompt 1", "Example prompt 2", ...]
# result = get_attributes(model_path, prompts)
# print(result)
