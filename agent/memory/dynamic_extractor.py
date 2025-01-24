import asyncio
from collections.abc import Callable
from typing import Any

from llama_index.core.async_utils import run_jobs
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from pydantic import Field

from agent.memory.memory_management.create_memory import create_memory, parse_input
from config import load_prompt_template

DEFAULT_ENTITY_TYPE: list[str] = [
    "PERSON",
    "ORGANIZATION",
    "PROJECT",
    "TASK",
    "DOCUMENT",
    "LOCATION",
    "EVENT",
    "GOAL",
    "IDEA",
    "THOUGHT",
    "EXPERIENCE",
    "DECISION",
    "RESOURCE",
    "FEEDBACK",
]

DEFAULT_RELATION_TYPE: list[str] = [
    "WORKS_ON",
    "PART_OF",
    "HAS_TASK",
    "LOCATED_IN",
    "MENTIONS",
    "CREATED_BY",
    "ASSIGNED_TO",
    "DEPENDS_ON",
    "RELATED_TO",
    "DOCUMENTS",
    "ACHIEVES",
    "INSPIRED_BY",
    "LEARNED_FROM",
    "REFLECTS",
    "EVOLVES_INTO",
    "DECISION_MADE",
    "USES_RESOURCE",
    "RESULTED_IN",
    "SUPERVISES",
]


class DynamicMemoryExtractor(TransformComponent):
    """
    DynamicLLMPathExtractor is a component for extracting structured information from text
    to build a knowledge graph. It uses an LLM to identify entities and their relationships,
    with the ability to infer entity types and expand upon an initial ontology.

    This extractor improves upon SimpleLLMPathExtractor by:
    1. Detecting entity types instead of labeling them generically as "entity" and "chunk".
    2. Accepting an initial ontology as input, specifying desired nodes and relationships.
    3. Encouraging ontology expansion through its prompt design.

    This extractor differs from SchemaLLMPathExtractor because:
    1. It interprets the passed possible entities and relations as an initial ontology.
    2. It encourages expansion of the initial ontology in the prompt.
    3. It aims for flexibility in knowledge graph construction while still providing guidance.

    Attributes:
        llm (LLM): The language model used for extraction.
        extract_prompt (PromptTemplate): The prompt template used to guide the LLM.
        parse_fn (Callable): Function to parse the LLM output into triplets.
        num_workers (int): Number of workers for parallel processing.
        max_triplets_per_chunk (int): Maximum number of triplets to extract per text chunk.
        allowed_entity_types (List[str]): List of initial entity types for the ontology.
        allowed_relation_types (List[str]): List of initial relation types for the ontology.
    """

    llm: LLM = Field(default=None)
    extract_prompt: PromptTemplate = Field(default=None)
    parse_fn: Callable = Field(default=parse_input)
    num_workers: int = Field(default=4)
    max_triplets_per_chunk: int = Field(default=10)
    allowed_entity_types: list[str] = Field(default_factory=lambda: DEFAULT_ENTITY_TYPE)
    allowed_relation_types: list[str] = Field(
        default_factory=lambda: DEFAULT_RELATION_TYPE
    )

    def __init__(
        self,
        llm: LLM | None = None,
        extract_prompt: PromptTemplate | None = None,
        max_triplets_per_chunk: int = 10,
        num_workers: int = 4,
        allowed_entity_types: list[str] | None = None,
        allowed_relation_types: list[str] | None = None,
    ) -> None:
        """
        Initialize the DynamicLLMPathExtractor.

        Args:
            llm (Optional[LLM]): The language model to use. If None, uses the default from Settings.
            extract_prompt (Optional[Union[str, PromptTemplate]]): The prompt template to use.
            parse_fn (Callable): Function to parse LLM output into triplets.
            max_triplets_per_chunk (int): Maximum number of triplets to extract per chunk.
            num_workers (int): Number of workers for parallel processing.
            allowed_entity_types (Optional[List[str]]): List of initial entity types for the ontology.
            allowed_relation_types (Optional[List[str]]): List of initial relation types for the ontology.
        """
        from llama_index.core import Settings

        from agent.tools.utils.get_schema import generate_json_schema

        super().__init__()
        self.llm = llm or Settings.llm
        self.extract_prompt = extract_prompt or load_prompt_template("property_graph")
        self.num_workers = num_workers
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.allowed_entity_types = allowed_entity_types or DEFAULT_ENTITY_TYPE
        self.allowed_relation_types = allowed_relation_types or DEFAULT_RELATION_TYPE
        self.synapse_tool_schema = generate_json_schema(create_memory)

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "DynamicLLMPathExtractor"

    def __call__(
        self, nodes: list[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        """
        Extract triples from nodes.

        Args:
            nodes (List[BaseNode]): List of nodes to process.
            show_progress (bool): Whether to show a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List[BaseNode]: Processed nodes with extracted information.
        """
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triplets asynchronously from a single node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        prompt = self.extract_prompt.format(
            max_knowledge_triplets=self.max_triplets_per_chunk,
            allowed_entity_types=self.allowed_entity_types,
            allowed_relation_types=self.allowed_relation_types,
        )
        examples = """
        Example Question:
        Please extract triples from the following text:
        [EXAMPLE INPUT]

        Example Response:
        ```json
        [
            {
                "source_entity": {
                "name": "alice_smith",
                "type": "PERSON",
                "label": "Alice Smith",
                "properties": {
                    "role": "Software Engineer",
                    "department": "AI Research"
                }
                },
                "relationship": {
                "source_id": "alice_smith",
                "target_id": "vision_project",
                "type": "WORKS_ON",
                "label": "Works on",
                "properties": {
                    "since": "2021-06"
                }
                },
                "target_entity": {
                "name": "vision_project",
                "type": "PROJECT",
                "label": "Computer Vision Project",
                "properties": {
                    "deadline": "2023-12-31",
                    "status": "Ongoing"
                }
                }
            }
        ]
        ```
        """
        try:
            response = self.llm.chat(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content=prompt + examples),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=" please extract triples from the following text: "
                        + text,
                    ),
                ],
                tools=[self.synapse_tool_schema],
            )
            llm_response: str = str(response.message.content)
            triplets = parse_input(llm_response)

            existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

            metadata = node.metadata.copy()
            for subj, rel, obj in triplets:
                subj.properties.update(metadata)
                obj.properties.update(metadata)
                rel.properties.update(metadata)

                existing_relations.append(rel)
                existing_nodes.append(subj)
                existing_nodes.append(obj)

            node.metadata[KG_NODES_KEY] = existing_nodes
            node.metadata[KG_RELATIONS_KEY] = existing_relations

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e

        return node

    async def acall(
        self, nodes: list[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        """
        Asynchronously extract triples from multiple nodes.

        Args:
            nodes (List[BaseNode]): List of nodes to process.
            show_progress (bool): Whether to show a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List[BaseNode]: Processed nodes with extracted information.
        """
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting and inferring knowledge graph from text",
        )
