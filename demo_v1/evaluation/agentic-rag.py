import asyncio
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph import run_graph, construct_graph


async def evaluate():
    graph_state = construct_graph()
    generation = await run_graph(graph_state)

    trace_file_name = f"agentic-rag-{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    trace_file_path = os.path.join("traces", trace_file_name)

    with open(trace_file_path, "w") as f:
        f.write(generation)
        print(f"Trace file saved as {trace_file_name}")


if __name__ == "__main__":
    asyncio.run(evaluate())
