import asyncio
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step


class EchoWorkflow(Workflow):
    """一个虚拟的工作流，只有一个步骤发送回给定的输入。"""

    @step()
    async def run_step(self, ev: StartEvent) -> StopEvent:
        message = str(ev.get("message", ""))
        return StopEvent(result=f"Message received: {message}")


# `echo_workflow` 会被LlamaDeploy导入
echo_workflow = EchoWorkflow()


async def main():
    print(await echo_workflow.run(message="Hello!"))


# 让这个脚本可以在shell中运行，这样我们就可以测试工作流的执行了
if __name__ == "__main__":
    asyncio.run(main())