"""Memory management module for the Brain agent system."""

# Standard library imports
from typing import Any

# import neo4j
# import neo4j.exceptions
from agent.agent import AgentState
from agent.message import Message, MessageState
from tools import FunctionCall, ToolCall

# Local imports
from .neo4j_property_graph import Neo4jPropertyGraphStore


class Hippocampus:
    """Central memory management system for the agent."""

    def __init__(self, agent_state: AgentState):
        """
        Initialize the Hippocampus with different memory components.

        Args:
            interface (AgentInterface): The interface for agent communication.
            agent_seed (int): Seed value for the agent.
        """
        self.agent_state = agent_state
        self.messages: list[Message] = [
            Message(
                role="system",
                content=self.agent_state.system_prompt,
                state=MessageState.SYSTEM_MESSAGE,
            )
        ]

        # date_string = datetime.now().strftime("the %d of %B, Year %Y, %I:%M in the %p")
        # if len(self.messages) > 1:
        #     self.append_to_messages(
        #         [
        #             Message(
        #                 role="system",
        #                 content="the interface was turned off.",
        #                 state=MessageState.SYSTEM_MESSAGE,
        #             ),
        #             Message(
        #                 role="system",
        #                 content=f"time has passed since the interface was last turned on. the interface is now on. the date is {date_string}.",
        #                 state=MessageState.SYSTEM_MESSAGE,
        #             ),
        #         ]
        #     )

    # def load_memory(self) -> None:
    #     """
    #     Load memory from the JSON file specified in the config.
    #     Parse the chat_history into Message types and add them to the hippocampus.
    #     """

    #     memory_path = os.path.join(self.agent_state.config.root_path, self.agent_state.config.memory_file)

    #     try:
    #         with open(memory_path) as f:
    #             memory_data = json.load(f)

    #         chat_history = memory_data.get('chat_history', [])
    #         for message_data in chat_history:
    #             old_message = self._construct_message(message_data)
    #             if old_message:
    #                 self.messages.append(old_message)

    #         self.agent_state.interface.console.print(f"[bold green]Loaded {len(self.messages)} messages from memory")

    #     except FileNotFoundError:
    #         self.agent_state.interface.console.print(f"[bold yellow]Warning: Memory file not found at {memory_path}")
    #     except json.JSONDecodeError:
    #         self.agent_state.interface.console.print(f"[bold red]Error: Invalid JSON in memory file {memory_path}")
    #     except Exception as e:
    #         self.interface.console.print(f"[bold red]Error loading memory: {e}")

    #     self.interface.console.print(f"[bold yellow]Loading graph memory for {self.agent_state.name}")
    #     self.get_memory_graph()

    #     if not self.graph_store:
    #         asyncio.run(self.interface.show_error_message(Exception("Graph store not initialized")))
            # return

        # storage_context = StorageContext.from_defaults(
        #     property_graph_store=self.graph_store,
        #     vector_store=self.graph_store,  # type: ignore reportArgumentType
        #     persist_dir=self.persist_dir,
        # )

        # try:
        #     index_id = self.graph_store._database if self.graph_store else self.agent_state.name.lower()
        #     index = load_index_from_storage(storage_context, index_id=index_id)
        # except Exception as e:
        #     asyncio.run(self.interface.show_error_message(e))
        #     return

        # if not isinstance(index, PropertyGraphIndex):
        #     raise ValueError("Index is not a PropertyGraphIndex")

        # memory_retriever = CypherRetriever(
        #     index.property_graph_store,
        #     llm=self.cortext
        # )

        # vector_retriever = VectorContextRetriever(
        #     graph_store=index.property_graph_store,
        #     index=index,
        #     similarity_top_k=5,
        #     path_depth=1
        # )

        # self.query_engine = index.as_query_engine(
        #     llm=self.cortext,
        #     sub_retrievers=[memory_retriever, vector_retriever],
        #     include_text=True,
        #     similarity_top_k=5,
        #     path_depth=1
        # )
        # self.interface.console.print("[bold green]Loaded graph memory")

    # def get_memory_graph(self):
    #     """
    #     Get the memory graph.
    #     """
    #     if not self.graph_store:
    #         try:
    #             if not self.config.neo4j_username or not self.config.neo4j_password or not self.config.neo4j_url:
    #                 raise neo4j.exceptions.ServiceUnavailable(
    #                     "Neo4j credentials not set, code Proxy.Memory.Config.Missing"
    #                 )
    #             self.graph_store = Neo4jPropertyGraphStore(
    #                 username=self.config.neo4j_username,
    #                 password=self.config.neo4j_password,
    #                 url=self.config.neo4j_url,
    #                 database=self.agent_state.name,
    #             )
    #         except neo4j.exceptions.ClientError as e:
    #             if e.code == "Neo.ClientError.Database.DatabaseNotFound":
    #                 print("Database not found, creating it")
    #                 self.create_memory_graph_database(self.agent_state.name.lower())
    #             elif e.code == "Neo.ClientError.Security.Unauthorized":
    #                 exception = Exception("Neo4j access was unauthorized. Please check your credentials in the memory.json config file.")
    #                 asyncio.run(self.agent_state.interface.show_error_message(exception))
    #                 return
    #             else:
    #                 raise e
    #         except neo4j.exceptions.ServiceUnavailable:
    #             # neo4j service is not running
    #             exception = Exception("Neo4j service is not running. Please start the service and try again")
    #             asyncio.run(self.agent_state.interface.show_error_message(exception))
    #             return
    #         except Exception as e:
    #             raise e
    #     return self.graph_store

    def create_memory_graph_database(self, database_name: str):
        """
        Create the memory graph database.
        """
        try:
            self.graph_store = Neo4jPropertyGraphStore(
                username="neo4j",
                password="password",
                url="bolt://localhost:7687",
            )
            self.graph_store.structured_query(f"CREATE DATABASE {database_name} IF NOT EXISTS WAIT 3 SECONDS")
            self.graph_store.structured_query(f"START DATABASE {database_name} WAIT 3 SECONDS")
            self.graph_store._database = database_name
        except Exception as e:
            raise e

    # def save_memory(self) -> None:
    #     """
    #     Save the current message list to a JSON file named {seed}-memory.json in the root_path.
    #     Create the memory file if it doesn't exist.
    #     """
    #     memory_filename = f"{self.agent_state.seed}-memory.json"
    #     memory_path = os.path.join(self.agent_state.config.root_path, memory_filename)

    #     # Ensure the directory exists
    #     os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    #     memory_data = {
    #         'chat_history': [message.to_dict() for message in set(self.messages)]
    #     }

    #     with open(memory_path, 'w') as f:
    #         json.dump(memory_data, f, indent=4)

    def _construct_message(self, message_data: dict[str, Any]) -> Message | None:
        """
        Construct a Message object from a dictionary of message data.

        Args:
            message_data (Dict[str, Any]): The dictionary containing message data.

        Returns:
            Optional[Message]: The constructed Message object or None if an error occurs.
        """
        try:
            # Extract basic fields
            role = message_data.get('role')
            content = message_data.get('content') or ""
            message_id = None
            state: MessageState
            if role == "assistant":
                state = MessageState.ASSISTANT_RESPONSE
            elif role == "user":
                state = MessageState.USER_INPUT
            elif role == "ipython" or role == "tool":
                message_id = message_data.get('tool_call_id') or message_data.get('tool_used_id')
                state = message_data.get("state", MessageState.SYSTEM_MESSAGE)
            elif role == "system":
                state = MessageState.SYSTEM_MESSAGE
            else:
                state = MessageState.ASSISTANT_RESPONSE

            # Handle tool_calls
            tool_calls = []
            if 'tool_calls' in message_data:
                for tool_call_data in message_data['tool_calls']:
                    tool_call_id = tool_call_data.get('id')
                    function_call = FunctionCall(
                        name=tool_call_data.get('name', ''),
                        arguments=tool_call_data.get('arguments', {})
                    )
                    tool_calls.append(ToolCall("function", function_call, tool_call_id))

            # Construct Message object
            message = Message(
                id=message_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                state=state,
                name=message_data.get('name')
            )
            return message
        except Exception as e:
            self.agent_state.interface.console.print(f"[bold red]Error constructing message: {e}")
            return None

    # async def get_documents(self):
    #     """
    #     Retrieve Discord messages and store them as documents.
    #     """
    #     message_count = 0
    #     with self.agent_state.interface.console.status("[bold orange]Getting Discord messages..."):
    #         try:
    #             self.documents, message_count = get_team_discord_messages()
    #         except Exception as e:
    #             self.agent_state.interface.console.print(f"[bold red]Error getting Discord messages: {e}")
    #             self.documents = []

        # await self.agent_state.interface(
        #     Message(
        #         role="system",
        #         content=f"Got {message_count} Discord messages"
        #     )
        # )
        # await self.agent_state.interface.display_memory_update(
        #     Message(
        #         role="system",
        #         content=f"First document:\n\n{self.documents[0].text}"
        #     )
        # )

    # async def setup_knowledge_graph(self):
    #     """
    #     Set up the knowledge graph.
    #     """
    #     try:
    #         await self.get_documents()
    #         await self._setup_knowledge_graph()
    #     except Exception as e:
    #         import traceback
    #         trace= traceback.format_exc()
    #         await self.interface.display_memory_update(
    #             Message(
    #                 role="system",
    #                 content=f"Error setting up knowledge graph: {e} \n\n{trace}"
    #             )
    #         )
    #         return

    # async def _setup_knowledge_graph(self):
    #     """
    #     Internal method to set up the knowledge graph.
    #     """

    #     self.get_memory_graph()

    #     if not self.graph_store:
    #         await self.interface.show_error_message(Exception("Graph store not initialized"))
    #         return

    #     storage_context = StorageContext.from_defaults(
    #         property_graph_store=self.graph_store,
    #         persist_dir=self.persist_dir
    #     )

    #     kg_extractor = ImplicitPathExtractor()
    #     dynamic_memory_extractor = DynamicMemoryExtractor(llm=self.cortext)

    #     property_graph_index: PropertyGraphIndex = PropertyGraphIndex.from_documents(
    #         llm=self.cortext,
    #         documents=self.documents[0:2],
    #         property_graph_store=self.graph_store,
    #         vector_store=self.graph_store,
    #         storage_context=storage_context,
    #         show_progress=True,
    #         kg_extractors=[kg_extractor, dynamic_memory_extractor]
    #     )

    #     vector_retriever = VectorContextRetriever(
    #         graph_store=property_graph_index.property_graph_store,
    #         index=property_graph_index,
    #         similarity_top_k=5,
    #         path_depth=1
    #     )

    #     memory_retriever = CypherRetriever(
    #         graph_store=property_graph_index.property_graph_store,
    #         llm=self.cortext
    #     )

    #     self.query_engine = property_graph_index.as_query_engine(
    #         llm=self.cortext,
    #         sub_retrievers=[memory_retriever, vector_retriever],
    #         include_text=True,
    #         similarity_top_k=5,
    #         path_depth=1
    #     )
    #     index_id = self.graph_store._database if self.graph_store else None
    #     property_graph_index.set_index_id(index_id or self.agent_state.name)
    #     property_graph_index.storage_context.persist(self.persist_dir)

    # def update_memories(self, step_result: StepResult):
    #     """
    #     Update all memory components based on the step result.

    #     Args:
    #         step_result (StepResult): The result of the step to update memories with.
    #     """
    #     if step_result.new_messages:
    #         self.append_to_messages(step_result.new_messages)
    #         self.save_memory()

    def get_context_for_next_step(self) -> str:
        """
        Generate context for the next step using all memory components.

        Returns:
            str: The generated context for the next step.
        """
        # core_summary = self.core_memory.get_recent_summary()
        # archival_context = self.archival_memory.search(core_summary, count=3)
        # recall_context = self.recall_memory.get_relevant_documents(core_summary)[:3]
        context = "your context is disabled for now"
        # context = f"Core Memory:\n{core_summary}\n\n"
        # context += "Relevant Archival Memories:\n" + "\n".join(archival_context) + "\n\n"
        # context += "Relevant Recall:\n" + "\n".join([doc.page_content for doc in recall_context])

        return context

    # def remember(self, query: str) -> str:
    #     """
    #     Query the memory to recall information.

    #     Args:
    #         query (str): The query to search in the memory.

    #     Returns:
    #         str: The result of the query.
    #     """
    #     if self.query_engine is None:
    #         return "No memory engine available!"
    #     try:
    #         results = self.query_engine.query(query)
    #         return str(results)
    #     except Exception as e:
    #         raise e

    def append_to_messages(self, added_messages: list[Message] | list[dict[str, Any]] | Message):
        """
        Append messages to the current message list.

        Args:
            added_messages (Union[List[Message], Dict[str, Any], List[Dict[str, Any]]]): The messages to append.
        """
        def append_constructed_message(message_dict: dict[str, Any]):
            constructed_message = self._construct_message(message_dict)
            if constructed_message:
                self.messages.append(constructed_message)

        if isinstance(added_messages, Message):
            self.messages.append(added_messages)
        elif isinstance(added_messages, list):
            for msg in added_messages:
                if isinstance(msg, dict):
                    append_constructed_message(msg)
                elif isinstance(msg, Message):
                    self.messages.append(msg)

    def get_message_by_id(self, message_id: str | None) -> Message | None:
        """
        Get a message by its ID.

        Args:
            message_id (str): The ID of the message to retrieve.

        Returns:
            Optional[Message]: The message with the specified ID or None if not found.
        """
        if not message_id:
            return None
        for message in self.messages:
            if message.id == message_id:
                return message
        return None

    def clear_messages(self):
        """
        Clear all messages from the current message list.
        """
        self.messages = []
