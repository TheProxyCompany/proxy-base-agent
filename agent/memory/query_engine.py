import re

from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from agent.api.chat_completion import get_chat_completion
from agent.api.message import Message
from config import MindConfig


class QueryEngine(CustomQueryEngine):
    graph_store: Neo4jPropertyGraphStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20
    mind_config: MindConfig

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        entities = self.get_entities(query_str)
        community_ids = self.retrieve_entity_communities(entities)
        # community_summaries = self.graph_store.get_community_summaries(self.mind_config)
        community_summaries = {}
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]
        return self.aggregate_answers(community_answers)

    def get_entities(self, query_str: str) -> list[str]:
        nodes_retrieved = self.index.as_retriever(similarity_top_k=self.similarity_top_k).retrieve(query_str)
        entities = set()
        pattern = r"(\w+(?:\s+\w+)*)\s*{[^}]*}{[^}]*}{[^}]*}\s*->\s*([^(]+?)\s*{[^}]*}{[^}]*}{[^}]*}\s*->\s*(\w+(?:\s+\w+)*)"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.DOTALL)
            for match in matches:
                entities.add(match[0])
                entities.add(match[2])

        return list(entities)

    def retrieve_entity_communities(self, entities: list[str]) -> list[int]:
        """Retrieve cluster information for given entities."""
        community_ids = []
        # for entity in entities:
        #     if entity in self.graph_store:
        #         community_ids.extend(self.graph_store.entity_info[entity])
        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        """Generate an answer from a community summary based on a given query using LLM."""
        system_message = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content="I need an answer based on the above information."),
        ]
        response = get_chat_completion(
            messages=messages,
            mind_config=self.mind_config
        )
        response_string = response.choices[0].message.content
        assert isinstance(response_string, str)
        return response_string

    def aggregate_answers(self, community_answers: list[str]) -> str:
        """Aggregate individual community answers into a final, coherent response."""
        system_message = "Combine the following intermediate answers into a final, concise response."
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=f"Intermediate answers: {community_answers}"),
        ]
        response = get_chat_completion(messages=messages, mind_config=self.mind_config)
        response_string = response.choices[0].message.content
        assert isinstance(response_string, str)
        return response_string
