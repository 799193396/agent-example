from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
import random
from llama_index.utils.workflow import draw_all_possible_flows


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str


class LoopEvent(Event):
    loop_output: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("坏事发生了")
            return LoopEvent(loop_output="回到第一步")
        else:
            print("今天发生了什么好事")
            return FirstEvent(first_output="第一步完成")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="第二步完成")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="完成流程")




async def main() -> None:
    w = MyWorkflow(timeout=10, verbose=True)
    result = await w.run(first_input="启动工作流")
    print(result)
    draw_all_possible_flows(MyWorkflow, "loop_work_flow.html")


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())