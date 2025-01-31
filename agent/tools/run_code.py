import ast
import contextlib
import io

from agent.agent import Agent
from agent.interaction import Interaction


def run_code(self: Agent, code: str) -> Interaction:
    """
    Executes Python code in a restricted environment and returns the output.
    Only uses Python's standard library - no external packages allowed.

    Safety features:
    - Captured stdout/stderr

    Args:
        code (str): The Python code to execute. Must use only standard library.
    """
    # Validate code structure using AST
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Prevent imports and dangerous built-ins
            if isinstance(node, ast.Import | ast.ImportFrom):
                raise ValueError("Import statements are not allowed")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile"]:
                        raise ValueError(f"Usage of {node.func.id}() is not allowed")
    except SyntaxError as e:
        return Interaction(
            role=Interaction.Role.TOOL,
            title=self.name + "'s code execution",
            content=f"Syntax Error: {e}",
            subtitle="Python Execution Error",
            color="red",
            emoji="warning",
        )

    # Capture output
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        with contextlib.redirect_stderr(output):
            try:
                exec(code, globals(), {})
                result = output.getvalue() or "Code executed successfully (no output)"
                color = "green"

            except Exception as e:
                result = f"Error during execution: {e}"
                color = "red"

    return Interaction(
        role=Interaction.Role.TOOL,
        title=self.name + "'s code execution",
        content=result,
        subtitle="Python Output",
        color=color,
        emoji="computer",
    )
