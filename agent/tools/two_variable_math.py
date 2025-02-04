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


def two_variable_math(
    self: Agent,
    number_1: float,
    number_2: float,
    operation: Operation,
) -> Interaction:
    """
    Perform a mathematical operation on two numbers.
    Do not use this tool for pointless operations.
    The results of this tool are always correct.

    Args:
        number_1 (number): The first number to perform the operation on.
        number_2 (number): The second number to perform the operation on.
        operation (Operation): The operation to perform between the two numbers.
    """
    operation = Operation(operation)

    match operation:
        case Operation.ADDITION:
            result = number_1 + number_2
        case Operation.SUBTRACTION:
            result = number_1 - number_2
        case Operation.MULTIPLICATION:
            result = number_1 * number_2
        case Operation.DIVISION:
            result = number_1 / number_2
        case Operation.GREATER_THAN:
            result = number_1 > number_2
        case Operation.LESS_THAN:
            result = number_1 < number_2
        case Operation.GREATER_THAN_OR_EQUAL_TO:
            result = number_1 >= number_2
        case Operation.LESS_THAN_OR_EQUAL_TO:
            result = number_1 <= number_2
        case Operation.POWER:
            result = number_1**number_2
        case _:
            raise ValueError(f"Invalid operation: {operation}")

    # If the operation yields a boolean result, display the comparison so that
    # the larger number comes first. This always shows a true statement.
    if isinstance(result, bool):
        if number_1 == number_2:
            content = f"{number_1} == {number_2} is True"
        else:
            larger = max(number_1, number_2)
            smaller = min(number_1, number_2)
            content = f"{larger} > {smaller} is True"
    else:
        content = f"{number_1} {operation.value} {number_2} = {result}"

    return Interaction(
        role=Interaction.Role.TOOL,
        title=self.name + "'s math",
        subtitle=f"{number_1} {operation.value} {number_2}",
        content=content,
        color="yellow",
        emoji="1234",
    )
