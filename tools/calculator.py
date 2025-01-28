import enum
import math

from agent.agent import Agent, AgentStatus
from agent.event import Event, EventState


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
    self: Agent,
    variable_a: float,
    variable_b: float,
    operation: MathOperation,
) -> Event:
    """
    Perform a mathematical operation on two variables.

    Args:
        variable_a (number): The first variable to perform the operation on. Integer or float.
        variable_b (number): The second variable to perform the operation on. Integer or float.
        operation (MathOperation): The operation to perform on the two variables.
    """
    self.status = AgentStatus.PROCESSING
    operation = MathOperation(operation)

    match operation:
        case MathOperation.ADDITION:
            result = variable_a + variable_b
        case MathOperation.SUBTRACTION:
            result = variable_a - variable_b
        case MathOperation.MULTIPLICATION:
            result = variable_a * variable_b
        case MathOperation.DIVISION:
            result = variable_a / variable_b
        case MathOperation.GREATER_THAN:
            result = variable_a > variable_b
        case MathOperation.LESS_THAN:
            result = variable_a < variable_b
        case MathOperation.GREATER_THAN_OR_EQUAL_TO:
            result = variable_a >= variable_b
        case MathOperation.LESS_THAN_OR_EQUAL_TO:
            result = variable_a <= variable_b
        case MathOperation.EQUAL_TO:
            result = variable_a == variable_b
        case MathOperation.NOT_EQUAL_TO:
            result = variable_a != variable_b
        case MathOperation.POWER:
            result = variable_a ** variable_b
        case MathOperation.ROOT:
            result = variable_a ** (1 / variable_b)
        case MathOperation.LOGARITHM:
            result = math.log(variable_a, variable_b)
        case MathOperation.EXPONENT:
            result = variable_a ** variable_b
        case MathOperation.FACTORIAL:
            result = math.factorial(int(variable_a))
        case _:
            self.status = AgentStatus.FAILED
            raise ValueError(f"Invalid operation: {operation}")

    content = f"{variable_a} {operation} {variable_b} = {result}"
    self.status = AgentStatus.SUCCESS
    return Event(
        state=EventState.TOOL,
        name=self.state.name + "'s calculator",
        content=content,
    )
