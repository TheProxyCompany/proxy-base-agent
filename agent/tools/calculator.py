import enum
import math

from agent.agent import Agent
from agent.interaction import Interaction


class MathOperation(enum.Enum):
    ADDITION = "add"
    SUBTRACTION = "subtract"
    MULTIPLICATION = "multiply"
    DIVISION = "divide"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL_TO = "greater_than_or_equal_to"
    LESS_THAN_OR_EQUAL_TO = "less_than_or_equal_to"
    EQUAL_TO = "equal_to"
    NOT_EQUAL_TO = "not_equal_to"
    POWER = "power"
    ROOT = "root"
    LOGARITHM = "logarithm"
    EXPONENT = "exponent"
    FACTORIAL = "factorial"

    def __str__(self):
        match self:
            case MathOperation.ADDITION:
                return "+"
            case MathOperation.SUBTRACTION:
                return "-"
            case MathOperation.MULTIPLICATION:
                return "*"
            case MathOperation.DIVISION:
                return "/"
            case MathOperation.GREATER_THAN:
                return ">"
            case MathOperation.LESS_THAN:
                return "<"
            case MathOperation.GREATER_THAN_OR_EQUAL_TO:
                return ">="
            case MathOperation.LESS_THAN_OR_EQUAL_TO:
                return "<="
            case MathOperation.EQUAL_TO:
                return "=="
            case MathOperation.NOT_EQUAL_TO:
                return "!="
            case MathOperation.POWER:
                return "**"
            case MathOperation.ROOT:
                return "root"
            case MathOperation.LOGARITHM:
                return "log"
            case MathOperation.EXPONENT:
                return "exp"
            case MathOperation.FACTORIAL:
                return "!"
            case _:
                return ""


def calculator(
    self: Agent, a: float, b: float, operation: MathOperation
) -> Interaction:
    """
    Perform a mathematical operation on two numbers.
    Response is formatted as "a `operation` b = result".
    Example: "1 + 2 = 3"

    Args:
        a (number): The first number to perform the operation on.
        b (number): The second number to perform the operation on.
        operation (MathOperation): The operation to perform with the two numbers.
    """
    self.status = Agent.Status.PROCESSING
    operation = MathOperation(operation)

    match operation:
        case MathOperation.ADDITION:
            result = a + b
        case MathOperation.SUBTRACTION:
            result = a - b
        case MathOperation.MULTIPLICATION:
            result = a * b
        case MathOperation.DIVISION:
            result = a / b
        case MathOperation.GREATER_THAN:
            result = a > b
        case MathOperation.LESS_THAN:
            result = a < b
        case MathOperation.GREATER_THAN_OR_EQUAL_TO:
            result = a >= b
        case MathOperation.LESS_THAN_OR_EQUAL_TO:
            result = a <= b
        case MathOperation.EQUAL_TO:
            result = a == b
        case MathOperation.NOT_EQUAL_TO:
            result = a != b
        case MathOperation.POWER:
            result = a**b
        case MathOperation.ROOT:
            result = a ** (1 / b)
        case MathOperation.LOGARITHM:
            result = math.log(a, b)
        case MathOperation.EXPONENT:
            result = a**b
        case MathOperation.FACTORIAL:
            result = math.factorial(int(a))
        case _:
            self.status = Agent.Status.FAILED
            raise ValueError(f"Invalid operation: {operation}")

    content = f"{a} {operation} {b} = {result}"
    self.status = Agent.Status.SUCCESS

    return Interaction(
        role=Interaction.Role.TOOL,
        name=self.name + "'s calculator",
        content=content,
    )
