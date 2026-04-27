from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from util import load_model


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class ProgressEvent(Event):
    msg: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="第一步开始"))
        return FirstEvent(first_output="第一步执行完成")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        llm, embed_model = load_model.get_llm()
        generator = await llm.astream_complete(
            "请帮我查一下西游记有多少章？"
        )
        async for response in generator:
            # 允许工作流对这段响应进行流处理
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return SecondEvent(
            second_output="第二步完成，并带有答案",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="第三步正在运行"))
        return StopEvent(result="工作流完成")


async def main():
    w = MyWorkflow(timeout=30, verbose=True)
    handler = w.run(first_input="启动工作流")

    # 通过stream_events方法，可以动态的看到每个事件里面的内容
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg)

    final_result = await handler
    print("结果", final_result)

    draw_all_possible_flows(MyWorkflow, filename="streaming_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())