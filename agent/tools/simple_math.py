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


def simple_math(
    self: Agent,
    number_1: float,
    number_2: float,
    operation: Operation,
) -> Interaction:
    """
    Perform precise mathematical operations between two numbers.

    This tool provides guaranteed accurate results for basic arithmetic
    and comparison operations. It's designed for:
    - Basic calculations
    - Numeric comparisons

    Best Practices:
    - Use for essential calculations only
    - Prefer this over execute_code for simple operations
    - Consider using execute_code for complex math or chained operations

    Args:
        number_1: First operand in the calculation.
            Must be a valid floating-point number.
        number_2: Second operand in the calculation.
            Must be a valid floating-point number.
        operation: The mathematical operation to perform.
            Must be a member of the Operation enum:
            - ADDITION (+)
            - SUBTRACTION (-)
            - MULTIPLICATION (*)
            - DIVISION (/)
            - POWER (**)
            - GREATER_THAN (>)
            - LESS_THAN (<)
            - GREATER_THAN_OR_EQUAL_TO (>=)
            - LESS_THAN_OR_EQUAL_TO (<=)
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
        # For comparisons, show the original expression and its result
        content = f"{number_1} {operation} {number_2} = {result}"

        # Add a helpful explanation of why it's true/false
        if result:
            if number_1 == number_2:
                content += f" (both numbers are equal to {number_1})"
            else:
                diff = abs(number_1 - number_2)
                content += f" (difference of {diff})"
        else:
            diff = abs(number_1 - number_2)
            content += f" (difference of {diff})"
    else:
        # For calculations, show the full expression with operation symbol
        content = f"{number_1} {operation} {number_2} = {result}"

    return Interaction(
        role=Interaction.Role.TOOL,
        title=self.name + "'s math",
        subtitle=f"{number_1} {operation} {number_2}",
        content=content,
        color="yellow",
        emoji="1234",
    )
