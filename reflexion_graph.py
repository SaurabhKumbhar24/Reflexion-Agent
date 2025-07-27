from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

graph = MessageGraph()

MAX_ITERATIONS = 2
DRAFT = 'draft'
REVISE = 'revise'
EXECUTE = 'execute'

graph.add_node(DRAFT, first_responder_chain)
graph.add_node(EXECUTE, execute_tools)
graph.add_node(REVISE, revisor_chain)

graph.add_edge(DRAFT, EXECUTE)
graph.add_edge(EXECUTE, REVISE)

def decide(state: List[BaseMessage]):
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXECUTE

graph.add_conditional_edges(REVISE, decide)

graph.set_entry_point(DRAFT)

app = graph.compile()


print(app.get_graph().draw_mermaid())

response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

print(response[-1].tool_calls[0]['args']['answer'])