from src.neo4j_rm import execute_cypher_queries, generate_cypher_queries
from src.preprocess import er_mapping, seggregate
from src.utils import run_rag
import src.generate_cypher as generate_cypher
import src.rag as rag

import json
def generate_prompts_neo4j(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    seggregate(data=data)
    er_mapping(data=data)
    print("here")
    relations = generate_cypher.generate()

    er = generate_cypher.entityRelations
    print("ER:",(relations))


    cypher_queries = generate_cypher_queries(relations=relations, er=er)

    execute_cypher_queries(cypher_queries)

    # print(rag.rag())
    r = rag.rag()
    print(r)
    return r