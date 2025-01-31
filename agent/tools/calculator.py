import enum

from agent.agent import Agent
from agent.interaction import Interaction


class Operation(enum.Enum):
    ADDITION = "add"
    SUBTRACTION = "subtract"
    MULTIPLICATION = "multiply"
    DIVISION = "divide"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL_TO = "greater_than_or_equal_to"
    LESS_THAN_OR_EQUAL_TO = "less_than_or_equal_to"
    POWER = "power"

    def __str__(self):
        match self:
            case Operation.ADDITION:
                return "+"
            case Operation.SUBTRACTION:
                return "-"
            case Operation.MULTIPLICATION:
                return "*"
            case Operation.DIVISION:
                return "/"
            case Operation.GREATER_THAN:
                return ">"
            case Operation.LESS_THAN:
                return "<"
            case Operation.GREATER_THAN_OR_EQUAL_TO:
                return ">="
            case Operation.LESS_THAN_OR_EQUAL_TO:
                return "<="
            case Operation.POWER:
                return "**"
            case _:
                return ""


def calculator(
    self: Agent, a: float, b: float, operation: Operation
) -> Interaction:
    """
    Perform a mathematical operation on two numbers.

    Args:
        a (number): The first number to perform the operation on.
        b (number): The second number to perform the operation on.
        operation (Operation): The operation to perform with the two numbers.
    """
    operation = Operation(operation)

    match operation:
        case Operation.ADDITION:
            result = a + b
        case Operation.SUBTRACTION:
            result = a - b
        case Operation.MULTIPLICATION:
            result = a * b
        case Operation.DIVISION:
            result = a / b
        case Operation.GREATER_THAN:
            result = a > b
        case Operation.LESS_THAN:
            result = a < b
        case Operation.GREATER_THAN_OR_EQUAL_TO:
            result = a >= b
        case Operation.LESS_THAN_OR_EQUAL_TO:
            result = a <= b
        case Operation.POWER:
            result = a**b
        case _:
            raise ValueError(f"Invalid operation: {operation}")

    content = f"{a} {operation} {b} = {result}"

    return Interaction(
        role=Interaction.Role.TOOL,
        title=self.name + "'s math",
        content=content,
        subtitle=operation.value,
        color="red",
        emoji="1234",
    )
