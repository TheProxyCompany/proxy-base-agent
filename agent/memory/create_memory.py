from pydantic import BaseModel

from agent.api import extract_content


class EntityModel(BaseModel):
    name: str
    label: str
    properties: dict[str, str]


class RelationModel(BaseModel):
    source_id: str
    target_id: str
    label: str
    properties: dict[str, str]


def create_memory(
    source_entity: EntityModel, relationship: RelationModel, target_entity: EntityModel
) -> tuple[EntityModel, RelationModel, EntityModel]:
    """
    Constructs a fundamental unit of knowledge by describing a source entity, a relationship, and a target entity. This sacred trio forms the basis for representing structured information within a memory management system.

    Args:
        source_entity (EntityModel): The source entity in the memory structure, which includes:
            - name (str): A unique identifier for the source entity.
            - label (str): A human-readable label that describes the source entity.
            - properties (Dict[str, str]): A dictionary of additional attributes or metadata associated with the source entity.

        relationship (RelationModel): The relationship connecting the source and target entities, which includes:
            - source_id (str): The identifier of the source entity involved in the relationship.
            - target_id (str): The identifier of the target entity involved in the relationship.
            - label (str): A label that describes the type or nature of the relationship.
            - properties (Dict[str, str]): A dictionary of additional attributes or metadata related to the relationship.

        target_entity (EntityModel): The target entity in the memory structure, which includes:
            - name (str): A unique identifier for the target entity.
            - label (str): A human-readable label that describes the target entity.
            - properties (Dict[str, str]): A dictionary of additional attributes or metadata associated with the target entity.

    Returns:
        Tuple[EntityModel, RelationModel, EntityModel]: A tuple representing the knowledge triplet (source_entity, relationship, target_entity), encapsulating the structured information.
    """
    return (source_entity, relationship, target_entity)


def parse_input(
    response: str, max_length: int = 128
) -> list[tuple[EntityNode, Relation, EntityNode]]:
    """
    Parse the LLM response into a list of triples.

    Args:
        response (str): The raw response from the LLM.
        max_length (int): Maximum length for entity names and relation labels.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of parsed triples.
    """
    triples = []
    _, tool_calls_data = extract_content(response)

    for tool_call in tool_calls_data:
        if tool_call.get("name") == "create_memory":
            params = tool_call.get("arguments", {})
            try:
                subject = EntityModel(**params.get("subject", {}))
                predicate = RelationModel(**params.get("predicate", {}))
                obj = EntityModel(**params.get("object", {}))
            except Exception as e:
                print(f"Error casting arguments to models: {e}")
                continue

            source = EntityNode(
                name=subject.name[:max_length],
                label=subject.label,
                properties=subject.properties,
            )
            target = EntityNode(
                name=obj.name[:max_length], label=obj.label, properties=obj.properties
            )
            relation = Relation(
                source_id=source.name,
                target_id=target.name,
                label=predicate.label[:max_length],
                properties=predicate.properties,
            )

            triples.append((source, relation, target))

    return triples
