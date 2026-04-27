from llama_deploy import (
    deploy_core,
    ControlPlaneConfig,
    SimpleMessageQueueConfig,
)


async def main():
    # This will run forever until you interrupt the process, like by pressing CTRL+C
    await deploy_core(
        control_plane_config=ControlPlaneConfig(),
        message_queue_config=SimpleMessageQueueConfig(),
    )

# 第一个启动的程序
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
