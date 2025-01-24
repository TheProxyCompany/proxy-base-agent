from agent.agent import Agent
from agent.message import Message, MessageState


def send_message(self: Agent, message: str) -> Message:
    """
    Conclude the sequence of tool usage by communicating the final, packaged result to the recipient.

    This function should be the last tool used during a sequence of tool use.
    It accumulates your past steps and communicates the packaged, markdown-formatted result of your work. Use this to deliver a synthesized and coherent response to the recipient.

    Psychological Frameworks Integrated:
    1. Summarization: Combines previous steps into a concise and coherent message.
    2. Closure: Marks the end of your reasoning process.
    3. Communication Theory: Focuses on effectively conveying information to the recipient.

    Arguments:
        message (str):
            The final message content to be sent to the recipient.
            This should be a packaged, markdown-formatted summary of the agent's work.
            Supports all Unicode characters, including emojis.
        inner_thoughts (str):
            The inner thoughts of the agent that led to the message.

    Returns:
        Message:
            The message that was sent, containing the final, formatted result.
    """

    self.should_handle_tool_result = False

    return Message(
        role="assistant",
        content=message,
        state=MessageState.ASSISTANT_RESPONSE,
        name=self.state.name + " sent a message",
    )
